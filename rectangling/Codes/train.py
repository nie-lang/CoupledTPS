import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
from network import build_model, CoupledTPS_RectanglingNet
from datetime import datetime
from dataset import TrainDataset
import glob
from loss import cal_perception_loss
import torchvision.models as models
import skimage
import torch.nn.functional as F




# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)

# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model')

# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # testing dataset
    test_data = TrainDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    # define the network
    net = CoupledTPS_RectanglingNet()
    vgg_model = models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        net = net.cuda()
        vgg_model = vgg_model.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)  # default as 0.97

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')



    print("##################start training#######################")
    print_interval = 300

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        total_loss_sigma = 0.
        perception_loss_sigma_list = [0.] * args.iter_num

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))



        #training
        for i, batch_value in enumerate(train_loader):

            input_tesnor = batch_value[0].float()
            mask_tesnor = batch_value[1].float()
            gt_tesnor = batch_value[2].float()

            if torch.cuda.is_available():
                input_tesnor = input_tesnor.cuda()
                mask_tesnor = mask_tesnor.cuda()
                gt_tesnor = gt_tesnor.cuda()

            mask_tesnor = torch.clamp(mask_tesnor, 0, 1)
            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = build_model(net, input_tesnor, mask_tesnor, args.iter_num)
            correction_list = batch_out['correction']
            # cal loss
            total_loss = 0
            perception_loss_list = []
            for k in range(args.iter_num):
                # perceptual loss
                perception_loss = cal_perception_loss(vgg_model, correction_list[k], gt_tesnor) * 1e-4
                perception_loss_list.append(perception_loss)
                total_loss = total_loss + perception_loss*(0.9**(args.iter_num-1-k))

            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()


            total_loss_sigma += total_loss.item()
            for k in range(args.iter_num):
                perception_loss_sigma_list[k] += perception_loss_list[k].item()


            print(glob_iter)
            # print loss etc.
            if i % print_interval == 0 and i != 0:
                total_loss_average = total_loss_sigma / print_interval
                perception_loss_average_list = [0.] * args.iter_num
                for k in range(args.iter_num):
                    perception_loss_average_list[k] = perception_loss_sigma_list[k]/ print_interval

                total_loss_sigma = 0.
                perception_loss_sigma_list = [0.] * args.iter_num


                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader), total_loss_average, optimizer.state_dict()['param_groups'][0]['lr']))

                # visualization
                writer.add_image("input", (input_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("mask", (mask_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("correction_last", (correction_list[-1][0]+1.)/2., glob_iter)
                writer.add_image("gt", (gt_tesnor[0]+1.)/2., glob_iter)

                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', total_loss_average, glob_iter)
                for k in range(args.iter_num):
                    writer.add_scalar('perception loss' + str(k), perception_loss_average_list[k], glob_iter)

            glob_iter += 1


        scheduler.step()
        # save model
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)

        # testing
        if (epoch+1)%1 == 0:
            loss2_list = [0.] * args.iter_num
            psnr_list = []
            ssim_list = []
            net.eval()
            for i, batch_value in enumerate(test_loader):

                input_tensor = batch_value[0].float()
                mask_tesnor = batch_value[1].float()
                gt_tensor = batch_value[2].float()

                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                    mask_tesnor = mask_tesnor.cuda()
                    gt_tensor = gt_tensor.cuda()

                mask_tesnor = torch.clamp(mask_tesnor, 0, 1)
                with torch.no_grad():
                    batch_out = build_model(net, input_tensor, mask_tesnor, args.iter_num)

                correction_list = batch_out['correction']

                # cal loss
                for k in range(args.iter_num):
                    loss2 = cal_appearance_loss(correction_list[k], gt_tensor)
                    loss2_list[k] += loss2.item()

                # choose the second iter's result to calculate PSNR/SSIM
                correction_tensor = correction_list[1]
                correction_np = ((correction_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
                gt_np = ((gt_tensor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)

                psnr = skimage.measure.compare_psnr(correction_np, gt_np, 255)
                ssim = skimage.measure.compare_ssim(correction_np, gt_np, data_range=255, multichannel=True)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                print(i)

            print("===================Results Analysis==================")
            print('average psnr:', np.mean(psnr_list))
            print('average ssim:', np.mean(ssim_list))
            print("##################end testing#######################")

            loss2_average_list = [0.] * args.iter_num
            for k in range(args.iter_num):
                loss2_average_list[k] = loss2_list[k]/665

            #writer.add_scalar('test_ave_loss1_vgg', ave_loss1, epoch+1)
            writer.add_scalar('test_ave_psnr', np.mean(psnr_list), epoch+1)
            writer.add_scalar('test_ave_ssim', np.mean(ssim_list), epoch+1)
            for k in range(args.iter_num):
                writer.add_scalar('test_ave_loss2_lp' + str(k), loss2_average_list[k], epoch+1)

            print("Testing: Epoch[{:0>3}/{:0>3}]   ave_loss1: {:.4f}  ".format(epoch + 1, args.max_epoch,  loss2_average_list[0]))


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--iter_num', type=int, default=4)

    parser.add_argument('--train_path', type=str, default='/opt/data/private/nl/Data/DIR-D/training/')
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/DIR-D/testing')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    print('<==================== jump into training function ===================>\n')
    train(args)


