import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.torch_tps_transform as torch_tps_transform
from torch.autograd import Variable
from torch.nn.modules import Module
import numpy as np




class Sobel_Loss(Module):
    def __init__(self):
        super(Sobel_Loss, self).__init__()
        x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32).reshape(1, 1, 3, 3)
        y = x.copy().T.reshape(1, 1, 3, 3)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        self.kernelx = Variable(x.contiguous())
        self.kernely = Variable(y.contiguous())
        self.criterion = torch.nn.L1Loss(reduction="mean")

    def forward(self, target, prediction, direction="x"):
        if direction == "x":
            tx = target
            px = prediction
            sobel_tx = F.conv2d(tx, self.kernelx, padding=1)
            sobel_px = F.conv2d(px, self.kernelx, padding=1)
            loss = self.criterion(sobel_tx, sobel_px)
        else:
            ty = target
            py = prediction
            sobel_ty = F.conv2d(ty, self.kernely, padding=1)
            sobel_py = F.conv2d(py, self.kernely, padding=1)
            loss = self.criterion(sobel_ty, sobel_py)

        return loss

def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):

    vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
    if torch.cuda.is_available():
        vgg_mean = vgg_mean.cuda()
    vgg_input = input_255-vgg_mean
    #x = vgg_model.features[0](vgg_input)
    #FeatureMap_list.append(x)

    x_list = []

    for i in range(0,layer_index+1):
        if i == 0:
            x = vgg_model.features[0](vgg_input)
        else:
            x = vgg_model.features[i](x)
            if i == 6 or i == 13 or i ==24:
                x_list.append(x)

    return x_list

def l_num_loss(img1, img2, l_num=1):

    loss = torch.mean(torch.abs((img1 - img2)**l_num))

    return loss



def cal_perception_loss_mask(vgg_model, rectangling, gt, mask):


    rectangling_feature_list = get_vgg19_FeatureMap(vgg_model, (rectangling+1)*127.5*mask, 24)
    gt_feature_list = get_vgg19_FeatureMap(vgg_model, (gt+1)*127.5*mask, 24)

    feature_loss_1 = l_num_loss(rectangling_feature_list[0], gt_feature_list[0], 2)
    feature_loss_2 = l_num_loss(rectangling_feature_list[1], gt_feature_list[1], 2)
    feature_loss_3 = l_num_loss(rectangling_feature_list[2], gt_feature_list[2], 2)

    feature_loss = (feature_loss_1 + feature_loss_2 + feature_loss_3)/3.

    return feature_loss

def cal_perception_loss(vgg_model, rectangling, gt):


    rectangling_feature_list = get_vgg19_FeatureMap(vgg_model, (rectangling+1)*127.5, 24)
    gt_feature_list = get_vgg19_FeatureMap(vgg_model, (gt+1)*127.5, 24)

    feature_loss_1 = l_num_loss(rectangling_feature_list[0], gt_feature_list[0], 2)
    feature_loss_2 = l_num_loss(rectangling_feature_list[1], gt_feature_list[1], 2)
    feature_loss_3 = l_num_loss(rectangling_feature_list[2], gt_feature_list[2], 2)

    feature_loss = (feature_loss_1 + feature_loss_2 + feature_loss_3)/3.

    return feature_loss
