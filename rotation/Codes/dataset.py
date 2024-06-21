from torch.utils.data import Dataset
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


class LabelDataset(Dataset):
    def __init__(self, datapath):

        self.width = 512
        self.height = 384

        self.train_path = datapath
        self.datas = OrderedDict()
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())


    def __getitem__(self, index):

        # load image1
        input = cv2.imread(self.datas['input']['image'][index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])

        # load image2
        gt = cv2.imread(self.datas['gt']['image'][index])
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt.astype(dtype=np.float32)
        gt = (gt / 127.5) - 1.0
        gt = np.transpose(gt, [2, 0, 1])

        # convert to tensor
        input_tensor = torch.tensor(input)
        gt_tensor = torch.tensor(gt)


        return (input_tensor, gt_tensor)

    def __len__(self):

        return len(self.datas['input']['image'])

class UnlabelDataset(Dataset):
    def __init__(self, datapath):

        self.width = 512
        self.height = 384

        self.train_path = datapath
        self.datas = OrderedDict()
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'input_aug' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())


    def __getitem__(self, index):

        # load image1
        input = cv2.imread(self.datas['input']['image'][index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])

        # load image2
        input_aug = cv2.imread(self.datas['input_aug']['image'][index])
        input_aug = cv2.resize(input_aug, (self.width, self.height))
        input_aug = input_aug.astype(dtype=np.float32)
        input_aug = (input_aug / 127.5) - 1.0
        input_aug = np.transpose(input_aug, [2, 0, 1])

        # convert to tensor
        input_tensor = torch.tensor(input)
        input_aug_tensor = torch.tensor(input_aug)


        return (input_tensor, input_aug_tensor)

    def __len__(self):

        return len(self.datas['input']['image'])


class DataProvider():

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.data_iter = None
        self.iter = 0
        self.epoch = 0
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last

    def build(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                 num_workers=self.num_workers,
                                                 pin_memory=self.pin_memory,
                                                 drop_last=self.drop_last)
        self.data_iter = _MultiProcessingDataLoaderIter(dataloader)

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iter += 1
            return batch

        except StopIteration:
            self.epoch += 1
            self.build()
            self.iter = 1
            batch = self.data_iter.next()
            return batch


# for testing
class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 384
        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'gt' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

    def __getitem__(self, index):

        # load image1
        input = cv2.imread(self.datas['input']['image'][index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])

        # load image2
        gt = cv2.imread(self.datas['gt']['image'][index])
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt.astype(dtype=np.float32)
        gt = (gt / 127.5) - 1.0
        gt = np.transpose(gt, [2, 0, 1])

        # convert to tensor
        input_tensor = torch.tensor(input)
        gt_tensor = torch.tensor(gt)

        return (input_tensor, gt_tensor)

    def __len__(self):

        return len(self.datas['input']['image'])


