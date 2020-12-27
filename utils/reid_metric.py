# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, fusion_feature=0, fusion_pid=0, cam_id=0, max_rank=50, feat_norm='yes', mode='no_fusion'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.fusion_feature = fusion_feature
        self.fusion_pid = fusion_pid
        self.fusion_camid = cam_id
        self.mode = mode

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        if self.mode == 'no_fusion':
            feats = torch.cat(self.feats, dim=0)
            # if self.feat_norm == 'yes':
            #     print("The test feature is normalized")
            #     feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            # query
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t()) # 1 is dismat self
            distmat = distmat.cpu().numpy()
            # a = []
            # b = []
            # c = []
            # for i in range(m):
            #     score = []
            #     new_id = []
            #     new_camid = []
            #     for idx, gid in enumerate(g_pids):
            #         index = (gid == g_pids) & (g_camids[idx] == g_camids)
            #         all_score = distmat[i][index]
            #         sum = all_score.sum()
            #         new_score = sum/len(all_score)
            #         if (new_score not in score):
            #             score.append(new_score)
            #             new_id.append(gid)
            #             new_camid.append(g_camids[idx])
            #         elif(new_score in score) and (gid not in new_id):
            #             score.append(new_score)
            #             new_id.append(gid)
            #             new_camid.append(g_camids[idx])
            #     a.append(score)
            #     b.append(new_id)
            #     c.append(new_camid)
            # sco = torch.cat(torch.tensor(a), dim=1)
            # distmat = np.asarray(sco)
            # g_pids = np.asarray(b[0])
            # g_camids = np.asarray(c[0])
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            return cmc, mAP
        elif self.mode =='fusion':
            feats = torch.cat(self.feats, dim=0)
            feats = torch.cat([feats, self.fusion_feature], dim=0)
            a = np.asarray(self.fusion_pid)
            self.pids.extend(list(a))
            b = np.asarray(self.fusion_camid)
            self.camids.extend(list(b))
            # if self.feat_norm == 'yes':
            #     print("The test feature is normalized")
            #     feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            # query
            feats = feats.detach()
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            return cmc, mAP
        elif self.mode == 'average':
            feats = torch.cat(self.feats, dim=0)
            feats = torch.cat([feats, self.fusion_feature], dim=0)
            a = np.asarray(self.fusion_pid)
            self.pids.extend(list(a))
            b = np.asarray(self.fusion_camid)
            self.camids.extend(list(b))
            # if self.feat_norm == 'yes':
            #     print("The test feature is normalized")
            #     feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            # query
            feats = feats.detach()
            qf = feats[:self.num_query]
            q_pids = np.asarray(self.pids[:self.num_query])
            q_camids = np.asarray(self.camids[:self.num_query])
            # gallery
            gf = feats[self.num_query:]
            g_pids = np.asarray(self.pids[self.num_query:])
            g_camids = np.asarray(self.camids[self.num_query:])
            m, n = qf.shape[0], gf.shape[0]
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, qf, gf.t())
            distmat = distmat.cpu().numpy()
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

            return cmc, mAP
        else:
            assert ('no fusion mode')



class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP