from torch.utils.data import Dataset
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random


# from imgaug import augmenters as iaa
# def data_aug(img):
#     oplist = []
#     if random.random() > 0.5:
#         oplist.append(iaa.GaussianBlur(sigma=(0.0, 1.0)))
#     elif random.random() > 0.5:
#         oplist.append(iaa.WithChannels(0, iaa.Add((1, 15))))
#     elif random.random() > 0.5:
#         oplist.append(iaa.WithChannels(1, iaa.Add((1, 15))))
#     elif random.random() > 0.5:
#         oplist.append(iaa.WithChannels(2, iaa.Add((1, 15))))
#     elif random.random() > 0.5:
#         oplist.append(iaa.AdditiveGaussianNoise(scale=(0, 10)))
#     elif random.random() > 0.5:
#         oplist.append(iaa.Sharpen(alpha=0.15))
#     elif random.random() > 0.5:
#         oplist.append(iaa.Clouds())
#     elif random.random() > 0.5:
#         oplist.append(iaa.Rain(speed=(0.1, 0.3)))

#     seq = iaa.Sequential(oplist)
#     images_aug = seq.augment_images([img])
#     return images_aug[0]


class TrainDataset():
    def __init__(self, data_dir):
        self.input_w = 512
        self.input_h = 384
        self.datas_infos = []

        self.datas_infos = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) \
                            if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith("p.jpg")]
        self.datas_num = len(self.datas_infos)
        print("number of datas:", self.datas_num)
        print("init data finished")

    def __len__(self):
        return self.datas_num

    def __getitem__(self, index):
        # Process the input image
        img_path = self.datas_infos[index]
        origimg = cv2.imread(img_path)

        img = cv2.resize(origimg, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        # img = data_aug(img)
        img = (img / 127.5) - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        # process the input mask
        img_mask_path = img_path.replace('.jpg', '_line_facemask.jpg')
        facemask = cv2.imread(img_mask_path, 0)  # read original mask
        facemask = cv2.resize(facemask,
                              (self.input_w , self.input_h),
                              interpolation=cv2.INTER_AREA)
        facemask = (facemask / 255.0)
        facemask = torch.from_numpy(facemask).float().unsqueeze(0)

        # Process the flow map
        img_map_x_path = img_path.replace('.jpg', '_ori2shape_mapx.exr')
        img_map_y_path = img_path.replace('.jpg', '_ori2shape_mapy.exr')

        flow_map_x = cv2.imread(img_map_x_path, cv2.IMREAD_ANYDEPTH)  # read flow map x direction
        flow_map_y = cv2.imread(img_map_y_path, cv2.IMREAD_ANYDEPTH)  # read flow map y direction
        flow_map_h, flow_map_w = flow_map_x.shape[:2]

        scale_x = self.input_w / flow_map_w
        scale_y = self.input_h / flow_map_h

        flow_map_x = cv2.resize(flow_map_x,
                                (self.input_w, self.input_h),
                                interpolation=cv2.INTER_AREA)
        flow_map_y = cv2.resize(flow_map_y,
                                (self.input_w, self.input_h),
                                interpolation=cv2.INTER_AREA)
        flow_map_x *= scale_x
        flow_map_y *= scale_y

        flow_map_x = flow_map_x[np.newaxis, :, :]
        flow_map_y = flow_map_y[np.newaxis, :, :]
        flow_map_x = torch.from_numpy(flow_map_x).float()
        flow_map_y = torch.from_numpy(flow_map_y).float()

        # Compute the weight
        mask_sum = torch.sum(facemask)
        weight = self.input_w * self.input_h / mask_sum - 1
        weight = torch.max(weight / 3, torch.ones(1))
        weight = weight.unsqueeze(-1).unsqueeze(-1)

        return img, flow_map_x, flow_map_y, facemask, weight


