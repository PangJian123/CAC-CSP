# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
# from data.datasets.bases import Dataset_new


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): # add attr in market.py, return attr in ImageDataset.__getitem__ 
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

# class ImageDataset_new(Dataset):
#     """A base class representing ImageDataset.
#
#     All other image datasets should subclass it.
#
#     ``__getitem__`` returns an image given index.
#     It will return ``img``, ``pid``, ``camid`` and ``img_path``
#     where ``img`` has shape (channel, height, width). As a result,
#     data in each batch has shape (batch_size, channel, height, width).
#     """
#
#     def __init__(self, train, query, gallery, **kwargs):
#         super(ImageDataset_new, self).__init__(train, query, gallery, **kwargs)
#
#     def __getitem__(self, index):
#         img_path, pid, camid = self.data[index]
#         img = read_image(img_path)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, pid, camid, img_path
#
#     def show_summary(self):
#         num_train_pids, num_train_cams = self.parse_data(self.train)
#         num_query_pids, num_query_cams = self.parse_data(self.query)
#         num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)
#
#         print('=> Loaded {}'.format(self.__class__.__name__))
#         print('  ----------------------------------------')
#         print('  subset   | # ids | # images | # cameras')
#         print('  ----------------------------------------')
#         print(
#             '  train    | {:5d} | {:8d} | {:9d}'.format(
#                 num_train_pids, len(self.train), num_train_cams
#             )
#         )
#         print(
#             '  query    | {:5d} | {:8d} | {:9d}'.format(
#                 num_query_pids, len(self.query), num_query_cams
#             )
#         )
#         print(
#             '  gallery  | {:5d} | {:8d} | {:9d}'.format(
#                 num_gallery_pids, len(self.gallery), num_gallery_cams
#             )
#         )
#         print('  ----------------------------------------')


