import torch
import torch.nn as nn
import torch.nn.functional as F

def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):

    #print(vgg_model)
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
    return torch.mean(torch.abs((img1 - img2)**l_num))


def cal_appearance_loss(rectangling, gt):

    pixel_loss = l_num_loss(rectangling, gt, 1)

    return pixel_loss

def cal_perception_loss(vgg_model, rectangling, gt):


    rectangling_feature_list = get_vgg19_FeatureMap(vgg_model, (rectangling+1)*127.5, 24)
    gt_feature_list = get_vgg19_FeatureMap(vgg_model, (gt+1)*127.5, 24)

    feature_loss_1 = l_num_loss(rectangling_feature_list[0], gt_feature_list[0], 2)
    feature_loss_2 = l_num_loss(rectangling_feature_list[1], gt_feature_list[1], 2)
    feature_loss_3 = l_num_loss(rectangling_feature_list[2], gt_feature_list[2], 2)

    feature_loss = (feature_loss_1 + feature_loss_2 + feature_loss_3)/3.

    return feature_loss
