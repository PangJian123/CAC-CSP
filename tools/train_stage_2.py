from __future__ import print_function, absolute_import
import os.path as osp
import os
import sys
import torch
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from config import PC_cfg as cfg
from data import make_data_loader, make_data_loader_target
from engine.trainer import create_supervised_evaluator
from utils.reid_metric import R1_mAP, R1_mAP_reranking
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from modeling.model import BaseModel
from utils.ckpt import load_checkpoint, save_checkpoint
from utils.logger import setup_logger
from  utils.logging import Logger
working_dir = osp.abspath(osp.join(osp.dirname("__file__"), osp.pardir))

def main(mode, ckpt, logger):
    logger.info(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    # ----load dataset------ #
    train_loader_s, _, _, num_classes = make_data_loader(cfg)
    train_loader_t, val_loader, num_query, _ = make_data_loader_target(cfg)
    cfg.DATASETS.NUM_CLASSES_S = num_classes
    pj_model = BaseModel(cfg)  # --------------
    # Evaluator
    if cfg.TEST.RE_RANKING == 'no':
        evaluator = create_supervised_evaluator(pj_model.Content_Encoder,
                                            metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device='cuda')
    else:
        evaluator = create_supervised_evaluator(pj_model.Content_Encoder,
                                            metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device='cuda')
    start_epoch = best_top1 = 0
    # Summary_writer
    writer = SummaryWriter()
    # Start training
    model_checkpoint = load_checkpoint(osp.join(working_dir, "CAC-CSP/logs/market_duke/market_duke_stage1.pth.tar"))
    pj_model.Content_Encoder.module.load_state_dict(model_checkpoint['Content_encoder'])
    pj_model.Content_optimizer.load_state_dict(model_checkpoint['Content_optimizer'])
    pj_model.Content_optimizer_fix.load_state_dict(model_checkpoint['Content_optimizer_fix'])
    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info("Validation Results - Epoch: {}".format(0))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    if mode == 'two':
        if cfg.DATASETS.NAMES == 'dukemtmc':
            rand_src_1 = np.asarray([0, 2, 4, 6])
            rand_src_2 = np.asarray([1, 3, 5, 7])
        elif cfg.DATASETS.NAMES == 'market1501':
            rand_src_1 = np.asarray([0, 1, 4])
            rand_src_2 = np.asarray([3, 2, 5])
        elif cfg.DATASETS.NAMES == 'msmt17':
            rand_src_1 = np.asarray([1, 3, 4, 5, 9, 11, 13])
            rand_src_2 = np.asarray([0, 2, 6, 7, 8, 10, 12, 14])
        for epoch in range(210, 240):
            pj_model.two_classifier(epoch, train_loader_s, train_loader_t, writer, logger, rand_src_1, rand_src_2)
            if (((epoch+1) % 1 == 0)):
                evaluator.run(val_loader)
                cmc, mAP = evaluator.state.metrics['r1_mAP']
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                is_best = cmc[0] > best_top1
                best_top1 = max(cmc[0], best_top1)
                save_checkpoint({
                'Content_encoder': pj_model.Content_Encoder.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, fpath=cfg.OUTPUT_DIR + 'checkpoint.pth.tar', info=ckpt+'.pth.tar')
                logger.info('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                    format(epoch, cmc[0], best_top1, ' *' if is_best else ''))
        writer.close()

if __name__ == '__main__':
    info = 'market_duke_stage2'
    sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, info+'.txt'))
    print(info)
    print('--------------------------------------------------')
    logger = setup_logger("reid_baseline", cfg.OUTPUT_DIR, 0, info+'.txt')
    main('two', info, logger)
