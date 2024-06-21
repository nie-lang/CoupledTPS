import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
#from torch_homography_model import build_model
from network import build_model, CoupledTPS_RotationNet
from datetime import datetime
from dataset import TestDataset, UnlabelDataset, LabelDataset, DataProvider
import glob
from loss import cal_appearance_loss, cal_perception_loss, cal_mutual_loss
import torchvision.models as models
import skimage


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

    # labeled dataset
    label_data = LabelDataset(args.train_path)
    label_loader = DataProvider(label_data, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size)
    # unlabeled dataset (training and testing)
    unlabel_data = UnlabelDataset(args.train_unlabel_path)
    unlabel_loader = DataProvider(unlabel_data, batch_size=int(args.batch_size/2), shuffle=True, num_workers=int(args.batch_size/2))

    # for the first 120 epochs, we train the network in the supervised way
    label_step = 1
    unlabel_step = 0
    batch_nums = int(5537 / args.batch_size / label_step)


    # testing dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    # define the network
    net = CoupledTPS_RotationNet()
    vgg_model = models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        net = net.cuda()
        vgg_model = vgg_model.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

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

        #input_tensor = 0
        print("start epoch {}".format(epoch))

        # when the epoch number exceeds 120, we train the network in the semi-supervised manner
        if epoch >= 120:
            label_step = 2
            unlabel_step = 1

        label_loss_sigma_list = [0.] * args.iter_num
        unlabel_loss_sigma = 0.


        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        net.train()
        # semi-supervised training
        for batch_idx in range(batch_nums):

            # training labeled data
            for i in range(label_step):
                # load data
                input_tensor, gt_tensor = label_loader.next()
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                    gt_tensor = gt_tensor.cuda()

                # forward, backward, update weights
                optimizer.zero_grad()

                batch_out = build_model(net, input_tensor, args.iter_num)
                correction_list = batch_out['correction']


                # cal loss
                total_loss = 0
                # perception_loss_list = []
                for k in range(args.iter_num):
                    perception_loss = cal_perception_loss(vgg_model, correction_list[k], gt_tensor)
                    perception_loss = perception_loss * 1e-4
                    label_loss_sigma_list[k] += perception_loss.item()
                    total_loss = total_loss + perception_loss*(0.9**(args.iter_num-1-k))


                total_loss.backward()
                # clip the gradient
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
                optimizer.step()

            # training unlabeled data via consistency contraint
            for i in range(unlabel_step):
                # load data
                input_tensor, input_aug_tensor = unlabel_loader.next()
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                    input_aug_tensor = input_aug_tensor.cuda()

                # forward, backward, update weights
                optimizer.zero_grad()

                batch_out = build_model(net, torch.cat([input_tensor, input_aug_tensor], 0), 1)
                norm_pre_mesh = batch_out['norm_pre_mesh_list'][0]

                total_loss = 0
                unlabel_loss = cal_mutual_loss(vgg_model, torch.cat([input_tensor, input_aug_tensor], 0), norm_pre_mesh)
                unlabel_loss_sigma += unlabel_loss.item()
                total_loss = total_loss + unlabel_loss

                total_loss.backward()
                # clip the gradient
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
                optimizer.step()


            if batch_idx % print_interval == 0 and batch_idx != 0:
                label_loss_average_list = [0.] * args.iter_num
                unlabel_loss_average = 0.
                for k in range(args.iter_num):
                    label_loss_average_list[k] = label_loss_sigma_list[k]/ print_interval/ label_step

                if unlabel_step == 0:
                    unlabel_loss_average = 0
                else:
                    unlabel_loss_average = unlabel_loss_sigma/ print_interval/ unlabel_step

                label_loss_sigma_list = [0.] * args.iter_num
                unlabel_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}]  Label Loss: {:.4f}  Unlabel Loss: {:.4f}".format(epoch + 1, args.max_epoch, label_loss_average_list[-1], unlabel_loss_average))

                # visualization
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                for k in range(args.iter_num):
                    writer.add_scalar('label loss' + str(k), label_loss_average_list[k], glob_iter)
                writer.add_scalar('unlabel loss', unlabel_loss_average, glob_iter)

            glob_iter += 1
            print(glob_iter)

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
            #loss1_list = []
            #loss2_list = []
            psnr_list = []
            ssim_list = []
            net.eval()
            for i, batch_value in enumerate(test_loader):

                input_tensor = batch_value[0].float()
                gt_tensor = batch_value[1].float()

                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                    gt_tensor = gt_tensor.cuda()

                with torch.no_grad():
                    batch_out = build_model(net, input_tensor, args.iter_num)

                correction_list = batch_out['correction']

                # cal loss
                #loss1 = cal_perception_loss(vgg_model, correction, gt_tesnor) * 1e-4
                for k in range(args.iter_num):
                    loss2 = cal_appearance_loss(correction_list[k], gt_tensor)
                    loss2_list[k] += loss2.item()

                # choose the second iter's result to calculate PSNR/SSIM
                correction_tensor = correction_list[2]
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

    #nl: create the argument parser
    parser = argparse.ArgumentParser()

    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=180)
    parser.add_argument('--iter_num', type=int, default=4)
    parser.add_argument('--train_path', type=str, default='/opt/data/private/nl/Data/DRC-D/training/')
    parser.add_argument('--train_unlabel_path', type=str, default='/opt/data/private/nl/Data/DRC-D/training_unlabel/')
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/DRC-D/testing/')

    #nl: parse the arguments
    args = parser.parse_args()
    print(args)

    print('<==================== jump into training function ===================>\n')
    #nl: rain
    train(args)


