import torch
from torch import nn
from torch.nn import functional as F
import time
import numpy as np
from solver import WarmupMultiStepLR
from utils.ckpt import AverageMeter
from layers.triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from modeling.baseline import Baseline
from tensorboardX import SummaryWriter
from modeling.nets import make_optimizer, print_network


class BaseModel(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._init_models()
        self._init_optimizers()

        print('---------- Networks initialized -------------')
        print_network(self.Content_Encoder)
        print('-----------------------------------------------')

    def _init_models(self):
        # -----------------Content_Encoder-------------------
        self.Content_Encoder = Baseline(self.cfg.DATASETS.NUM_CLASSES_S, 1, self.cfg.MODEL.PRETRAIN_PATH, 'bnneck',
                                      'after', self.cfg.MODEL.NAME, 'imagenet')
        # -----------------Criterion----------------- #
        self.xent = CrossEntropyLabelSmooth(num_classes=self.cfg.DATASETS.NUM_CLASSES_S).cuda()
        self.triplet = TripletLoss(0.3)
        self.Smooth_L1_loss = torch.nn.SmoothL1Loss(reduction='mean').cuda()
        # --------------------Cuda------------------- #
        self.Content_Encoder = torch.nn.DataParallel(self.Content_Encoder).cuda()

    def _init_optimizers(self):
        self.Content_optimizer = make_optimizer(self.cfg, self.Content_Encoder)
        self.Content_optimizer_fix = make_optimizer(self.cfg, self.Content_Encoder, fix=True)
        self.scheduler = WarmupMultiStepLR(self.Content_optimizer, (30, 55), 0.1, 1.0 / 3,
                                           500, "linear")
        self.scheduler_fix = WarmupMultiStepLR(self.Content_optimizer_fix, (30, 55), 0.1, 1.0 / 3,
                                           500, "linear")
        self.schedulers = []
        self.optimizers = []

    def reset_model_status(self):
        self.Content_Encoder.train()

    def two_classifier(self, epoch, train_loader_s, train_loader_t, writer, logger, rand_src_1, rand_src_2,
                       print_freq=1):
        self.reset_model_status()
        self.epoch = epoch
        self.scheduler.step(epoch)
        self.scheduler_fix.step(epoch)
        target_iter = iter(train_loader_t)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        if (epoch < 80) or (110 <= epoch < 170):
            mode = 'normal_c1_c2'
        elif (80 <= epoch < 110) or (170 <= epoch < 210):
            mode = 'reverse_c1_c2'
        elif 210 <= epoch:
            mode = 'fix_c1_c2'
        for i, inputs in enumerate(train_loader_s):
            data_time.update(time.time() - end)
            try:
                inputs_target = next(target_iter)
            except:
                target_iter = iter(train_loader_t)
                inputs_target = next(target_iter)
            img_s, pid_s, camid_s = self._parse_data(inputs)
            img_t, pid_t, camid_t = self._parse_data(inputs_target)
            content_code_s, content_feat_s = self.Content_Encoder(img_s)
            pid_s_12 = np.asarray(pid_s.cpu())
            camid_s = np.asarray(camid_s.cpu())
            idx = []
            for c_id in rand_src_1:
                if len(np.where(c_id == camid_s)[0]) == 0:
                    continue
                else:
                    idx.append(np.where(c_id == camid_s)[0])
            if idx == [] or len(idx[0]) == 1:
                idx = [np.asarray([a]) for a in range(self.cfg.SOLVER.IMS_PER_BATCH)]
            idx = np.concatenate(idx)
            pid_1 = torch.tensor(pid_s_12[idx]).cuda()
            feat_1 = content_feat_s[idx]
            idx = []
            for c_id in rand_src_2:
                if len(np.where(c_id == camid_s)[0]) == 0:
                    continue
                else:
                    idx.append(np.where(c_id == camid_s)[0])
            if idx == [] or len(idx[0]) == 1:
                idx = [np.asarray([a]) for a in range(self.cfg.SOLVER.IMS_PER_BATCH)]
            idx = np.concatenate(idx)
            pid_2 = torch.tensor(pid_s_12[idx]).cuda()
            feat_2 = content_feat_s[idx]
            if mode == 'normal_c1_c2':
                class_1 = self.Content_Encoder(feat_1, mode='c1')
                class_2 = self.Content_Encoder(feat_2, mode='c2')
                ID_loss_1 = self.xent(class_1, pid_1)
                ID_loss_2 = self.xent(class_2, pid_2)
                ID_tri_loss = self.triplet(content_feat_s, pid_s)
                total_loss = ID_loss_1 + ID_loss_2 + ID_tri_loss[0]
                self.Content_optimizer.zero_grad()
                total_loss.backward()
                self.Content_optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % print_freq == 0:
                    logger.info('Epoch: [{}][{}/{}]\t'
                                'Time {:.3f} ({:.3f})\t'
                                'Data {:.3f} ({:.3f})\t'
                                'ID_loss: {:.3f}  ID_loss_1: {:.3f}  ID_loss_2: {:.3f}   tri_loss: {:.3f} '
                                .format(epoch, i + 1, len(train_loader_s),
                                        batch_time.val, batch_time.avg,
                                        data_time.val, data_time.avg,
                                        total_loss.item(), ID_loss_1.item(), ID_loss_2.item(), ID_tri_loss[0].item()
                                        ))
            elif mode == 'reverse_c1_c2':
                class_1 = self.Content_Encoder(feat_1, mode='c2')
                class_2 = self.Content_Encoder(feat_2, mode='c1')
                ID_loss_1 = self.xent(class_1, pid_1)
                ID_loss_2 = self.xent(class_2, pid_2)
                ID_tri_loss = self.triplet(content_feat_s, pid_s)
                total_loss = ID_loss_1 + ID_loss_2 + ID_tri_loss[0]
                self.Content_optimizer_fix.zero_grad()
                total_loss.backward()
                self.Content_optimizer_fix.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % print_freq == 0:
                    logger.info('Epoch: [{}][{}/{}]\t'
                                'Time {:.3f} ({:.3f})\t'
                                'Data {:.3f} ({:.3f})\t'
                                'ID_loss: {:.3f}  ID_loss_1: {:.3f}  ID_loss_2: {:.3f}   tri_loss: {:.3f}'
                                .format(epoch, i + 1, len(train_loader_s),
                                        batch_time.val, batch_time.avg,
                                        data_time.val, data_time.avg,
                                        total_loss.item(), ID_loss_1.item(), ID_loss_2.item(), ID_tri_loss[0].item()
                                        ))
            elif mode == 'fix_c1_c2':
                class_1 = self.Content_Encoder(feat_1, mode='c2')
                class_2 = self.Content_Encoder(feat_2, mode='c1')
                ID_loss_1 = self.xent(class_1, pid_1)
                ID_loss_2 = self.xent(class_2, pid_2)

                content_code_t, content_feat_t = self.Content_Encoder(img_t)
                tar_class_1 = self.Content_Encoder(content_feat_t, mode='c1')
                tar_class_2 = self.Content_Encoder(content_feat_t, mode='c2')
                tar_L1_loss = self.Smooth_L1_loss(tar_class_1, tar_class_2)
                ID_tri_loss = self.triplet(content_feat_s, pid_s)
                arg_c1 = torch.argmax(tar_class_1, dim=1)
                arg_c2 = torch.argmax(tar_class_2, dim=1)
                arg_idx = []
                fake_id = []
                for i_dx, data in enumerate(arg_c1):
                    if (data == arg_c2[i_dx]) and (((tar_class_1[i_dx][data] + tar_class_2[i_dx][arg_c2[i_dx]])/2) > 0.8):
                        arg_idx.append(i_dx)
                        fake_id.append(data)
                if 210 <= epoch < 220:
                    if arg_idx != []:
                        ID_loss_fake = self.xent(content_code_t[arg_idx], torch.tensor(fake_id).cuda())
                        total_loss = ID_loss_1 + ID_loss_2 + 0.5 * tar_L1_loss + ID_tri_loss[0]
                    else:
                        ID_loss_fake = torch.tensor([0])
                        total_loss = ID_loss_1 + ID_loss_2 + 0.5 * tar_L1_loss + ID_tri_loss[0]
                if 220 <= epoch:
                    if arg_idx != []:
                        ID_loss_fake = self.xent(content_code_t[arg_idx], torch.tensor(fake_id).cuda())
                        total_loss = ID_loss_1 + ID_loss_2 + 0.08 * ID_loss_fake + ID_tri_loss[0] + 0.5 * tar_L1_loss
                    else:
                        ID_loss_fake = torch.tensor([0])
                        total_loss = ID_loss_1 + ID_loss_2 + ID_tri_loss[0] + 0.5 * tar_L1_loss

                self.Content_optimizer_fix.zero_grad()
                total_loss.backward()
                self.Content_optimizer_fix.step()
                batch_time.update(time.time() - end)
                end = time.time()
                if (i + 1) % print_freq == 0:
                    logger.info('Epoch: [{}][{}/{}]\t'
                                'Time {:.3f} ({:.3f})\t'
                                'Data {:.3f} ({:.3f})\t'
                                'ID_loss: {:.3f}  ID_loss_1: {:.3f}  ID_loss_2: {:.3f}  tar_L1_loss: {:.3f}  tri_loss: {:.3f}  ID_loss_fake:  {:.6f}'
                                .format(epoch, i + 1, len(train_loader_s),
                                        batch_time.val, batch_time.avg,
                                        data_time.val, data_time.avg,
                                        total_loss.item(), ID_loss_1.item(), ID_loss_2.item(), tar_L1_loss.item(),
                                        ID_tri_loss[0].item(), ID_loss_fake.item()))
    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        camids = camids.cuda()
        return inputs, targets, camids




