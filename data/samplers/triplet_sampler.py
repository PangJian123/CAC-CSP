# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

class Gallery_Sampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, num_Cams_T):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_cams = num_Cams_T  # cam_duke = 8
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.index_dict_cam = defaultdict(list)
        for index, (_, pid, camid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)   # {} pid:index
            self.index_dict_cam[index].append(camid)  # { } index:camid
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        batch_idxs = []
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            camid = [self.index_dict_cam[c][0] for c in idxs]  # camid:[1,1,2,3,2,1,4,5,1]
            for i in range(self.num_cams):
                if camid.count(i) < self.num_instances:
                    continue
                else:
                    camid_arr = np.array(camid)
                    idxs_arr = np.array(idxs)
                    same_cam_idx = np.where(camid_arr == i)  # [0,1,5,8]
                    currant_idx = idxs_arr[same_cam_idx]
                    random.shuffle(currant_idx)
                    for idx in currant_idx:
                        batch_idxs.append(idx)
                        if len(batch_idxs) == self.num_instances:
                            batch_idxs_dict[pid].append(batch_idxs)   # defaultdict(<class 'list'>, {1626: [[1817, 2587, 16877, 10488], [15887, 3110, 1574, 5170], [14220, 7797, 5965, 0]]})
                            batch_idxs = []
                    batch_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
                    continue
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

class CamSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.index_dict_cam = defaultdict(list)
        for index, (_, pid, camid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)  # {} pid:index
            self.index_dict_cam[index].append(camid)     # { } index:camid
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])  # idxs of one person
            # idxs = np.random.choice(idxs, size=2, replace=True)
            random.shuffle(idxs)
            for i, idx_currant in enumerate(idxs):
                for idx in idxs[i+1:]:
                    if self.index_dict_cam[idx_currant]==self.index_dict_cam[idx]:
                        continue
                    else:
                        batch_idxs_dict[pid].append([idx_currant, idx])

        avai_pids = copy.deepcopy(self.pids)
        batch_idxs = []
        final_idxs = []
        while len(avai_pids) >= self.batch_size:
            selected_pids = random.sample(avai_pids, self.batch_size)  # 随机挑batch_size个pid
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        self.length = len(final_idxs)
        return iter(final_idxs)

        #     if len(idxs) < self.num_instances:
        #         idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
        #     random.shuffle(idxs)
        #     idxs_new = []
        #     camid1 = [-1]
        #     for i, idx in enumerate(idxs):
        #         camid2 = self.index_dict_cam[idx]
        #         if camid1[0]==camid2[0]:
        #             continue
        #         else:
        #             idxs_new.append(idx)
        #             if (i+1)%2==0:
        #                 camid1 = [-1]
        #             else:
        #                 camid1 = self.index_dict_cam[idx]
        #     camid_new = [self.index_dict_cam[idx] for idx in idxs_new]
        #     batch_idxs = []
        #     for idx in idxs_new:
        #         batch_idxs.append(idx) #
        #         if len(batch_idxs) == self.num_instances:
        #             batch_idxs_dict[pid].append(batch_idxs)     #  batch_idxs_dict{165:[51,53,16,97]...} (pid:165,img index:[51,53..])
        #             batch_idxs = []
        #
        # avai_pids = copy.deepcopy(self.pids)
        # final_idxs = []
        #
        # while len(avai_pids) >= self.num_pids_per_batch:
        #     # for i in avai_pids:
        #     #     if len(batch_idxs_dict[i])==0:
        #     #         continue
        #     selected_pids = random.sample(avai_pids, self.num_pids_per_batch) # num_pids_per_batch=batch_size/num_instances,16组
        #     for pid in selected_pids:
        #         if len(batch_idxs_dict[pid]) == 0: #　由于相机采集标准，缺少部分行人样本
        #             avai_pids.remove(pid)
        #             continue
        #         batch_idxs = batch_idxs_dict[pid].pop(0) # batch_idxs=[51,53,16,97] ,batch_idxs_dict通过pop(0)逐渐减少其中的Index
        #         final_idxs.extend(batch_idxs)            #　batch_idxs_dict当没有某一个label的图片时，avai_pids--
        #         if len(batch_idxs_dict[pid]) == 0:
        #             avai_pids.remove(pid)
        #
        # self.length = len(final_idxs)
        # return iter(final_idxs)    # final_idxs是4个为一组的索引的列表[51,53,16,97,...]四个是一张行人图片

    def __len__(self):
        return self.length
# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
