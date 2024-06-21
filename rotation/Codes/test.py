# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, CoupledTPS_RotationNet
from dataset import *
import os
import numpy as np
import skimage
import cv2


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    #nl: set num_workers = the number of cpus
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = CoupledTPS_RotationNet()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    if args.model_path != '':
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(args.model_path))
    else:
        print('No checkpoint found!')



    print("##################start testing#######################")
    psnr_list = []
    ssim_list = []
    net.eval()
    for i, batch_value in enumerate(test_loader):

        input_tesnor = batch_value[0].float()
        gt_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            input_tesnor = input_tesnor.cuda()
            gt_tesnor = gt_tesnor.cuda()

        with torch.no_grad():
            batch_out = build_model(net, input_tesnor, args.iter_num)
        correction_list = batch_out['correction']
        correction_final = correction_list[-1]

        correction_np = ((correction_final[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        input_np = ((input_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        gt_np = ((gt_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)

        # calculate psnr/ssim
        psnr = skimage.measure.compare_psnr(correction_np, gt_np, 255)
        ssim = skimage.measure.compare_ssim(correction_np, gt_np, data_range=255, multichannel=True)

        if not os.path.exists("../result/"):
            os.makedirs("../result/")
        path = "../result/" + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, correction_np)


        print('i = {}, psnr = {:.6f}'.format( i+1, psnr))

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        torch.cuda.empty_cache()

    print("===================Results Analysis==================")
    print('average psnr:', np.mean(psnr_list))
    print('average ssim:', np.mean(ssim_list))
    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iter_num', type=int, default=3)
    parser.add_argument('--model_path', type=str, default='../model/rotation.pth')
    parser.add_argument('--test_path', type=str, default='../DRC-D/testing/')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
