import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
from network import build_model, CoupledTPS_PortraitNet, warp_with_flow
from datetime import datetime
from dataset import TrainDataset
import glob
from loss import cal_perception_loss_mask, Sobel_Loss, l_num_loss
import torchvision.models as models
import skimage
import json


from tqdm import tqdm

eps = 1e-6
# -----------------Using the model to output the flow map for the distortion image------------------
def estimation_flowmap(net, img, iter):
    net.eval()
    img = cv2.resize(img, (512, 384))
    img = (img / 127.5) - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    # img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0).cuda()
    with torch.no_grad():
        batch_out = build_model(net, img, iter)
    output = batch_out['flow_list'][-1].detach().cpu().squeeze(0).numpy()
    return output


# ----------------------The computation process of face metric ---------------------------------------
def compute_cosin_similarity(preds, gts):
    people_num = gts.shape[0]
    points_num = gts.shape[1]
    similarity_list = []
    preds = preds.astype(np.float32)
    gts = gts.astype(np.float32)
    for people_index in range(people_num):
        # the index 63 of lmk is the center point of the face, that is, the tip of the nose
        pred_center = preds[people_index, 63, :]
        pred = preds[people_index, :, :]
        pred = pred - pred_center[None, :]
        gt_center = gts[people_index, 63, :]
        gt = gts[people_index, :, :]
        gt = gt - gt_center[None, :]

        dot = np.sum((pred * gt), axis=1)
        pred = np.sqrt(np.sum(pred * pred, axis=1))
        gt = np.sqrt(np.sum(gt * gt, axis=1))

        similarity_list_tmp = []
        for i in range(points_num):
            if i != 63:
                similarity = (dot[i] / (pred[i] * gt[i] + eps))
                similarity_list_tmp.append(similarity)

        similarity_list.append(np.mean(similarity_list_tmp))

    return np.mean(similarity_list)


# --------------------The normalization function -----------------------------------------------------
def normalization(x):
    return [(float(i) - min(x)) / float(max(x) - min(x) + eps) for i in x]


# -------------------The computation process of line metric-------------------------------------------
def compute_line_slope_difference(pred_line, gt_k):
    scores = []
    for i in range(pred_line.shape[0] - 1):
        pk = (pred_line[i + 1, 1] - pred_line[i, 1]) / (pred_line[i + 1, 0] - pred_line[i, 0] + eps)
        score = np.abs(pk - gt_k)
        scores.append(score)
    scores_norm = normalization(scores)
    score = np.mean(scores_norm)
    score = 1 - score
    return score


# -------------------------------Compute the out put flow map -------------------------------------------------
def compute_ori2shape_face_line_metric(model, oriimg_paths, iter):
    line_all_sum_pred = []
    face_all_sum_pred = []
    for oriimg_path in tqdm(oriimg_paths):
        # Get the [Source image]
        ori_img = cv2.imread(oriimg_path)  # Read the oriinal image
        ori_height, ori_width, _ = ori_img.shape  # get the size of the oriinal image
        input = ori_img.copy()  # get the image as the input of our model

        # Get the [flow map]"""
        pred = estimation_flowmap(model, input, iter)
        pflow = pred.transpose(1, 2, 0)
        predflow_x, predflow_y = pflow[:, :, 0], pflow[:, :, 1]

        scale_x = ori_width / predflow_x.shape[1]
        scale_y = ori_height / predflow_x.shape[0]
        predflow_x = cv2.resize(predflow_x, (ori_width, ori_height)) * scale_x
        predflow_y = cv2.resize(predflow_y, (ori_width, ori_height)) * scale_y

        # Get the [predicted image]"""
        ys, xs = np.mgrid[:ori_height, :ori_width]
        mesh_x = predflow_x.astype("float32") + xs.astype("float32")
        mesh_y = predflow_y.astype("float32") + ys.astype("float32")
        pred_out = cv2.remap(input, mesh_x, mesh_y, cv2.INTER_LINEAR)
        cv2.imwrite(oriimg_path.replace(".jpg", "_pred.jpg"), pred_out)

        # Get the landmarks from the [gt image]
        stereo_lmk_file = open(oriimg_path.replace(".jpg", "_stereo_landmark.json"))
        stereo_lmk = np.array(json.load(stereo_lmk_file), dtype="float32")

        # Get the landmarks from the [source image]
        ori_lmk_file = open(oriimg_path.replace(".jpg", "_landmark.json"))
        ori_lmk = np.array(json.load(ori_lmk_file), dtype="float32")

        # Get the landmarks from the the pred out
        out_lmk = np.zeros_like(ori_lmk)
        for i in range(ori_lmk.shape[0]):
            for j in range(ori_lmk.shape[1]):
                x = ori_lmk[i, j, 0]
                y = ori_lmk[i, j, 1]
                if y < predflow_y.shape[0] and x < predflow_y.shape[1]:
                    out_lmk[i, j, 0] = x - predflow_x[int(y), int(x)]
                    out_lmk[i, j, 1] = y - predflow_y[int(y), int(x)]
                else:
                    out_lmk[i, j, 0] = x
                    out_lmk[i, j, 1] = y

        # Compute the face metric
        face_pred_sim = compute_cosin_similarity(out_lmk, stereo_lmk)
        face_all_sum_pred.append(face_pred_sim)
        stereo_lmk_file.close()
        ori_lmk_file.close()

        # Get the line from the [gt image]
        gt_line_file = oriimg_path.replace(".jpg", "_line_lines.json")
        lines = json.load(open(gt_line_file))

        # Get the line from the [source image]
        ori_line_file = oriimg_path.replace(".jpg", "_lines.json")
        ori_lines = json.load(open(ori_line_file))

        # Get the line from the pred out
        pred_ori2shape_lines = []
        for index, ori_line in enumerate(ori_lines):
            ori_line = np.array(ori_line, dtype="float32")
            pred_ori2shape = np.zeros_like(ori_line)
            for i in range(ori_line.shape[0]):
                x = ori_line[i, 0]
                y = ori_line[i, 1]
                pred_ori2shape[i, 0] = x - predflow_x[int(y), int(x)]
                pred_ori2shape[i, 1] = y - predflow_y[int(y), int(x)]
            pred_ori2shape = pred_ori2shape.tolist()
            pred_ori2shape_lines.append(pred_ori2shape)

        # Compute the lines score
        line_pred_ori2shape_sum = []
        for index, line in enumerate(lines):
            gt_line = np.array(line, dtype="float32")
            pred_ori2shape = np.array(pred_ori2shape_lines[index], dtype="float32")
            gt_k = (gt_line[1, 1] - gt_line[0, 1]) / (gt_line[1, 0] - gt_line[0, 0] + eps)
            pred_ori2shape_score = compute_line_slope_difference(pred_ori2shape, gt_k)
            line_pred_ori2shape_sum.append(pred_ori2shape_score)
        line_all_sum_pred.append(np.mean(line_pred_ori2shape_sum))

    return np.mean(line_all_sum_pred) * 100, np.mean(face_all_sum_pred) * 100


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
    train_data = TrainDataset(data_dir=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)


    # define the network
    net = CoupledTPS_PortraitNet()#build_model(args.model_name)
    vgg_model = models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        net = net.cuda()
        vgg_model = vgg_model.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    # sobel loss
    sobel_criterion = Sobel_Loss().cuda()

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
    max_score1 = 0
    max_score2 = 0

    input_tensorboard = 0
    correction_tensorboard = 0
    gt_tensorboard = 0

    for epoch in range(start_epoch, args.max_epoch):

        #input_tensor = 0
        print("start epoch {}".format(epoch))


        #total_loss_sigma = 0.
        perception_loss_sigma_list = [0.] * args.iter_num
        flow_loss_sigma_list = [0.] * args.iter_num
        sobel_loss_sigma_list = [0.] * args.iter_num


        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        net.train()

        #training
        for batch_idx, batch_value in enumerate(train_loader):

            img = batch_value[0].float()
            flow_map_x = batch_value[1].float()
            flow_map_y = batch_value[2].float()
            facemask = batch_value[3].float()
            weight = batch_value[4].float()

            if torch.cuda.is_available():
                img = img.cuda()
                flow_map_x = flow_map_x.cuda()
                flow_map_y = flow_map_y.cuda()
                facemask = facemask.cuda()
                weight = weight.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()
            batch_out = build_model(net, img, args.iter_num)
            correction_list = batch_out['correction']
            flow_list = batch_out['flow_list']

            # gt correction
            gt_tensor = warp_with_flow(img, torch.cat([flow_map_x, flow_map_y], 1))
            #
            input_tensorboard = img
            correction_tensorboard = correction_list[-1]
            gt_tensorboard = gt_tensor

            # get the mask
            mask = (facemask * (weight - 1)) + 1
            if epoch >= 70:
                mask = (facemask * (weight - 1))*2 + 1

            # cal loss
            total_loss = 0
            # perception_loss_list = []
            for k in range(args.iter_num):
                # flow L1 loss
                flow_loss = l_num_loss(flow_list[k][:,0,:,:].unsqueeze(1)*mask, flow_map_x*mask, 1) + l_num_loss(flow_list[k][:,1,:,:].unsqueeze(1)*mask, flow_map_y*mask, 1)
                # Sobel L1 loss
                sobel_loss = sobel_criterion(flow_map_x*mask, flow_list[k][:,0,:,:].unsqueeze(1)*mask, direction='x') +sobel_criterion(flow_map_y*mask, flow_list[k][:,1,:,:].unsqueeze(1)*mask, direction='y')
                # perceptual loss
                perception_loss = cal_perception_loss_mask(vgg_model, correction_list[k], gt_tensor, mask)
                perception_loss = perception_loss * 2e-4

                # sum
                loss = perception_loss + flow_loss + sobel_loss * 10.
                flow_loss_sigma_list[k] += flow_loss.item()
                sobel_loss_sigma_list[k] += sobel_loss.item()
                perception_loss_sigma_list[k] += perception_loss.item()
                total_loss = total_loss + loss*(0.9**(args.iter_num-1-k))

            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, norm_type=2)
            optimizer.step()


            if batch_idx % print_interval == 0 and batch_idx != 0:
                perception_loss_average_list = [0.] * args.iter_num
                flow_loss_average_list = [0.] * args.iter_num
                sobel_loss_average_list = [0.] * args.iter_num

                for k in range(args.iter_num):
                    perception_loss_average_list[k] = perception_loss_sigma_list[k]/ print_interval
                    sobel_loss_average_list[k] = sobel_loss_sigma_list[k]/ print_interval
                    flow_loss_average_list[k] = flow_loss_sigma_list[k]/ print_interval


                perception_loss_sigma_list = [0.] * args.iter_num
                sobel_loss_sigma_list = [0.] * args.iter_num
                flow_loss_sigma_list = [0.] * args.iter_num

                print("Training: Epoch[{:0>3}/{:0>3}]  Perception Loss: {:.4f}".format(epoch + 1, args.max_epoch, perception_loss_average_list[0]))

                # # visualization
                writer.add_image("input", (input_tensorboard[0]+1.)/2., glob_iter)
                writer.add_image("correction_last", (correction_tensorboard[0]+1.)/2., glob_iter)
                writer.add_image("gt", (gt_tensorboard[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)

                writer.add_scalar('flow loss', flow_loss_average_list[1], glob_iter)
                writer.add_scalar('sobel loss', sobel_loss_average_list[1], glob_iter)
                for k in range(args.iter_num):
                    writer.add_scalar('perception loss' + str(k), perception_loss_average_list[k], glob_iter)


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
            # iter 1
            oriimg_paths = []
            for root, dirs, files in os.walk(args.test_path):
                for file_name in files:
                    if file_name.endswith(".jpg"):
                        if "line" not in file_name and "stereo" not in file_name and "pred" not in file_name:
                            oriimg_paths.append(os.path.join(root, file_name))
            print("The number of images: :", len(oriimg_paths))
            print("--------------------------Test--------------------------")
            line_score, face_score = compute_ori2shape_face_line_metric(net, oriimg_paths, 1)
            print("Line_score1 = {:.3f}, Face_score1 = {:.3f} ".format(line_score, face_score))
            writer.add_scalar('Line_score1', line_score, epoch+1)
            writer.add_scalar('Face_score1', face_score, epoch+1)

            # save the best model
            if line_score >= max_score1:
                # update max_score
                max_score1 = line_score
                # save model
                filename = 'best_model_iter1.pth'
                model_save_path = os.path.join(MODEL_DIR, filename)
                state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
                torch.save(state, model_save_path)
                # add scalar to tensorboard
                writer.add_scalar('best_model_score1', line_score, epoch+1)

            # iter 2
            oriimg_paths = []
            for root, dirs, files in os.walk(args.test_path):
                for file_name in files:
                    if file_name.endswith(".jpg"):
                        if "line" not in file_name and "stereo" not in file_name and "pred" not in file_name:
                            oriimg_paths.append(os.path.join(root, file_name))
            print("The number of images: :", len(oriimg_paths))
            print("--------------------------Test--------------------------")
            line_score, face_score = compute_ori2shape_face_line_metric(net, oriimg_paths, 2)
            print("Line_score = {:.3f}, Face_score = {:.3f} ".format(line_score, face_score))
            writer.add_scalar('Line_score2', line_score, epoch+1)
            writer.add_scalar('Face_score2', face_score, epoch+1)

            # save the best model
            if line_score >= max_score2:
                # update max_score
                max_score2 = line_score
                # save model
                # filename ='epoch' + str(epoch+1).zfill(3) + '_best_model.pth'
                filename = 'best_model_iter2.pth'
                model_save_path = os.path.join(MODEL_DIR, filename)
                state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
                torch.save(state, model_save_path)
                # add scalar to tensorboard
                writer.add_scalar('best_model_score2', line_score, epoch+1)




if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--iter_num', type=int, default=4)
    parser.add_argument('--train_path', type=str, default='/opt/data/private/nl/Data/PortraitData/train_4_3/')
    parser.add_argument('--test_path', type=str, default='/opt/data/private/nl/Data/PortraitData/test/')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    print('<==================== jump into training function ===================>\n')
    # rain
    train(args)


