import torch
import torch.nn as nn
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps2flow as torch_tps2flow
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as T

# we define the control points as (7+1) * (10+1)
# but we discard some points near the image center (refer to `get_rigid_mesh()' function)
grid_h = 7
grid_w = 10


# warping the img using flow (backward flow)
def warp_with_flow(img, flow):
    #initilize grid_coord
    batch, C, H, W = img.shape
    coords0 = torch.meshgrid(torch.arange(H).cuda(), torch.arange(W).cuda())
    coords0 = torch.stack(coords0[::-1], dim=0).float()
    coords0 = coords0[None].repeat(batch, 1, 1, 1)  # bs, 2, h, w

    # target coordinates
    target_coord = coords0 + flow

    # normalization
    target_coord_w = target_coord[:,0,:,:]*2./float(W) - 1.
    target_coord_h = target_coord[:,1,:,:]*2./float(H) - 1.
    target_coord_wh = torch.stack([target_coord_w, target_coord_h], 1)

    #
    warped_img = F.grid_sample(img, target_coord_wh.permute(0,2,3,1), align_corners=True)

    return warped_img

'''
def get_rigid_mesh(batch_size, height, width):


    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt
'''


def get_rigid_mesh(batch_size, height, width):


    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2

    # to discard some points
    grid_index = (ori_pt[:, :, 1] >=(192-75) ) & (ori_pt[:, :, 1] <= (192+75))
    grid_index = grid_index & (ori_pt[:, :, 0] >= (256-100)) & (ori_pt[:, :, 0] <= (256+100))
    grid_index = ~grid_index
    grid = ori_pt[grid_index]


    grid = grid.unsqueeze(0).expand(batch_size, -1, -1)

    return grid

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 2) # bs*N*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2


def build_model(net, input_tensor, iter_num):
    batch_size, _, img_h, img_w = input_tensor.size()


    flow_list, norm_pre_mesh_list = net(input_tensor, iter_num)

    correction_list = []

    for i in range(iter_num):
        # warp tilted image
        warped_img = warp_with_flow(input_tensor, flow_list[i])
        # list appending
        correction_list.append(warped_img)


    out_dict = {}
    out_dict.update(correction = correction_list, flow_list = flow_list, norm_pre_mesh_list=norm_pre_mesh_list)


    return out_dict




def get_res18_FeatureMap(resnet18_model):
    # suppose input resolution to be   512*384
    # the output resolution should be  32*24

    layers_list = []

    layers_list.append(resnet18_model.conv1)    #stride 2*2     H/2
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4

    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8
    layers_list.append(resnet18_model.layer3)                  #H/16

    feature_extractor = nn.Sequential(*layers_list)
    #exit(0)

    return feature_extractor



# define and forward
class CoupledTPS_PortraitNet(nn.Module):

    def __init__(self):
        super(CoupledTPS_PortraitNet, self).__init__()


        # suppose the input to be 32*24
        # the output resolution fould be 4*3
        self.regressNet_part1 = nn.Sequential(

            # input:  h/16 * w/16
            # 32*24
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 16*12

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 8*6

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            # 4*3
        )

        self.regressNet_part2 = nn.Sequential(

            # input resolution: 4*3*512
            nn.Linear(in_features=6144, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=1024, out_features=82*2, bias=True)
        )


        # kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        # define the res18 backbone
        resnet18_model = models.resnet.resnet18(pretrained=True)
        #print(resnet18_model)
        if torch.cuda.is_available():
            resnet18_model = resnet18_model.cuda()
        self.feature_extractor = get_res18_FeatureMap(resnet18_model)

    # forward
    def forward(self, input_tesnor, iter_num):
        batch_size, _, img_h, img_w = input_tesnor.size()


        # feature extraction
        feature_ori = self.feature_extractor(input_tesnor)
        feature = feature_ori.clone()

        flow = 0
        flow_list = []
        norm_pre_mesh_list = []
        for i in range(iter_num):
            # estimate the TPS motions for control points
            temp = self.regressNet_part1(feature).contiguous()
            temp = temp.view(temp.size()[0], -1)
            offset = self.regressNet_part2(temp)
            mesh_motion = offset.reshape(-1, 82, 2)

            # convert TPS deformation to optical flows (image resolution: 384*512)
            rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
            norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
            pre_mesh = rigid_mesh + mesh_motion
            norm_pre_mesh = get_norm_mesh(pre_mesh, img_h, img_w)

            delta_flow = torch_tps2flow.transformer(input_tesnor, norm_rigid_mesh, norm_pre_mesh, (img_h, img_w))
            # note: the optical flow is backward flow ----   2*384*512

            # compute the current flow (delta_flow "+" flow)
            if i == 0:
                flow = delta_flow + flow
            else:
                # warp the flow using delta_flow
                warped_flow = warp_with_flow(flow, delta_flow)
                flow = delta_flow + warped_flow
            # save flow
            flow_list.append(flow)
            norm_pre_mesh_list.append(norm_pre_mesh)


            if i < iter_num-1:
                _, _, fea_h, fea_w = feature.size()

                # downsample the optical flow
                down_flow = F.interpolate(flow, size=(fea_h, fea_w), mode='bilinear', align_corners=True)
                down_flow_w = down_flow[:,0,:,:]*fea_w/img_w
                down_flow_h = down_flow[:,1,:,:]*fea_h/img_h
                down_flow = torch.stack([down_flow_w, down_flow_h], 1)

                # warp features
                feature = warp_with_flow(feature_ori, down_flow)


        return flow_list, norm_pre_mesh_list
