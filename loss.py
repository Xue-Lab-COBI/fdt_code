# Created by Renzhi He, UC Davis, 2023

"""
Training losses for FDT refractive-index tomography. Provides the main Loss
module that combines an L1/L2-blended data-fidelity term between rendered and
measured intensity images, an SSIM structural term, and total-variation (TV)
regularization in the lateral (xy) and axial (z) directions, plus auxiliary
components: a VGG16-based perceptual loss, a channel diversity loss, and a
(legacy) DnCNN denoiser-prior regularizer.

FDT 折射率层析重建的训练损失模块。核心 Loss 类组合了渲染强度与测量强度之间的
L1/L2 混合数据保真项、SSIM 结构相似性项，以及横向（xy）和轴向（z）的全变分
（TV）正则项；此外还提供辅助组件：基于 VGG16 的感知损失、通道多样性损失，
以及（历史遗留的）DnCNN 去噪先验正则项。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#import skimage
#from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
#from absl import flags
import logging

from ssim import SSIM
#from .dncnn import DnCNN

# get total number of visible gpus
NUM_GPUS = torch.cuda.device_count()


########################################
###       Tensorboard & Helper       ###
########################################
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.transforms import functional as F

class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss: compares deep feature maps of prediction and target
    instead of raw pixels. Uses the first layers of a frozen, pretrained VGG16 as the
    feature extractor (weights are not updated during training).

    基于 VGG16 的感知损失：比较预测图与目标图的深层特征图而非原始像素。使用冻结的
    预训练 VGG16 前几层作为特征提取器（训练过程中不更新其权重）。
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:10]  # 使用VGG的前23层
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        # EN: VGG expects 3-channel images: replicate single-channel inputs across RGB,
        # then take the MSE between the extracted VGG feature maps.
        # 中文：VGG 需要三通道输入：将单通道图像复制到 RGB 三个通道后提取 VGG 特征图，
        # 再对特征图计算 MSE。
        if input.shape[1] != 3:
            input = input.repeat(3, 1, 1, 1).transpose(1,0)[20]
            target = target.repeat(3, 1, 1, 1).transpose(1,0)[20]
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = nn.functional.mse_loss(input_features, target_features)
        return loss

# 假设input_stack和target_stack是你的两个图像堆栈，形状为(N, H, W)
# 需要对它们进行适当的预处理（缩放、归一化等）



# EN: Write a scalar value to a TensorBoard summary writer and flush immediately.
# 中文：向 TensorBoard 写入一个标量并立即刷新缓冲区。
def record_summary(writer, name, value, step):
    writer.add_scalar(name, value, step)
    writer.flush()


# EN: The following reshape_image_* helpers coerce tensors of various ranks into the
# 4D (batch, H, W, channel) layout used by the losses.
# 中文：以下 reshape_image_* 辅助函数将不同维数的张量统一整形为损失函数所用的
# 4 维（批次, 高, 宽, 通道）布局。
def reshape_image(image):
    if len(image.shape) == 2:
        image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    elif len(image.shape) == 3:
        image_reshaped = image.unsqueeze(-1)
    else:
        image_reshaped = image
    return image_reshaped


def reshape_image_2(image):
    image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    return image_reshaped


def reshape_image_3(image):
    image_reshaped = image.unsqueeze(-1)
    return image_reshaped


def reshape_image_5(image):
    shape = image.shape
    image_reshaped = image.view(-1, shape[2], shape[3], 1)
    return image_reshaped


#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TVLoss(nn.Module):
    """
    Anisotropic-normalized 2D total-variation loss for 4D tensors (N, C, H, W): sums the
    squared differences of neighboring pixels along height and width, normalizes by the
    number of comparisons, and averages over the batch.

    针对 4 维张量 (N, C, H, W) 的二维全变分损失：分别对高、宽方向相邻像素差的平方
    求和，按比较次数归一化后再对批次取平均。
    """
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
# EN: Penalize similarity between z-layers: average off-diagonal cosine similarity of the
# flattened, L2-normalized channels — lower means more diverse layers.
# 中文：惩罚各 z 层之间的相似性：对展平并 L2 归一化的通道计算余弦相似度矩阵的
# 非对角线均值——数值越低表示各层差异越大。
def diversity_loss(tensor):
    """
    Compute a diversity loss for a tensor of shape (H, W, C) to encourage
    different features in each channel/layer.

    Parameters:
    tensor (torch.Tensor): Input tensor of shape (H, W, C).

    Returns:
    torch.Tensor: Computed diversity loss.
    """
    H, W, C = tensor.shape

    # Reshape tensor to (C, H*W) and normalize
    tensor_flat = tensor.permute(2, 0, 1).reshape(C, H*W)
    tensor_norm = F.normalize(tensor_flat, p=2, dim=1)

    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(tensor_norm, tensor_norm.t())

    # Since we want diversity, we are interested in minimizing the similarity.
    # Diagonal elements are self-similarities, so we exclude them.
    loss = (similarity_matrix.sum() - similarity_matrix.trace()) / (C * (C - 1))
    return loss
# EN: Lateral (in-plane) anisotropic L1 total variation for an (H, W, N) volume, applied
# to every z-slice; used as the xy smoothness regularizer on the RI volume.
# 中文：对 (H, W, N) 体数据逐 z 层计算的横向（面内）各向异性 L1 全变分；用作 RI 体
# 在 xy 方向的平滑正则项。
def tv_loss(x):
    """
    Compute the Total Variation Loss for a 3D stack (H, W, N).

    Parameters:
    x (torch.Tensor): Input tensor of shape (H, W, N).

    Returns:
    torch.Tensor: Total Variation Loss.
    """
    # Calculate the difference in the horizontal direction
    horizontal_diff = torch.abs(x[:, :-1, :] - x[:, 1:, :])

    # Calculate the difference in the vertical direction
    vertical_diff = torch.abs(x[:-1, :, :] - x[1:, :, :])

    # Sum up the differences
    loss = horizontal_diff.sum() + vertical_diff.sum()
    return loss

if __name__ == '__main__':
    pass
    #main()


class Loss(nn.Module):
    """
    Main training loss for FDT. Combines a data-fidelity term between rendered
    intensities Hxhat and measurements y (L1, L2, or an L1/L2 blend that anneals from L2
    to L1 over the first training steps), an SSIM structural term, and total-variation
    regularization on the reconstructed RI volume xhat in both xy and z. Optional legacy
    components (DnCNN denoiser prior, perceptual and diversity losses) are kept but
    disabled by default.

    FDT 的主训练损失。组合以下几项：渲染强度 Hxhat 与测量值 y 之间的数据保真项
    （L1、L2，或在训练初期从 L2 逐渐退火到 L1 的混合项）、SSIM 结构相似性项，以及
    对重建 RI 体 xhat 在 xy 与 z 方向的全变分正则项。可选的历史遗留组件（DnCNN
    去噪先验、感知损失与多样性损失）予以保留，但默认关闭。
    """
    def __init__(self, DnCNNN_channels=1, tower_idx=None, Hreal=None, Himag=None):
        super(Loss, self).__init__()
        self.tower_idx = tower_idx
        self.Hreal = Hreal
        self.Himag = Himag
        self.SSIM = SSIM()
        self.TVLoss = TVLoss()

        # model_path = os.path.join('./dncnn/model_zoo', 'dncnn_15' + '.pth')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # from dncnn.models.network_dncnn import DnCNN as net
        # model = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
        # model.load_state_dict(torch.load(model_path), strict=True)
        # model.eval()
        # for k, v in model.named_parameters():
        #     v.requires_grad = False
        # self.DnCNN = model.to(device)
        # # dncnn
        # num_of_layers = 17
        # logdir = "./model/dncnn_logs/DnCNN-S-25"
        # net = DnCNN(channels=DnCNNN_channels, num_of_layers=num_of_layers)
        # device_ids = [0]
        # model = nn.DataParallel(net, device_ids=device_ids).cuda()
        # model.load_state_dict(torch.load(os.path.join(logdir, 'net.pth')), strict=False)
        # # print('load DnCNN Done...')
        # # lock the parameters of DnCNN
        # for name, parameter in model.named_parameters():
        #     parameter.requires_grad = False
        # self.dncnn = model

        # Setup parameters

    ##############################
    ###     Loss Functions     ###
    ##############################

    def forward(self, args, Hxhat, xhat, y, steps, xhat_gt=None,tower_idx=0, reuse=False):
        """
        Compute the total loss and its individual components. Hxhat: rendered intensity,
        xhat: reconstructed RI volume, y: measured intensity, steps: current iteration
        (drives loss-schedule annealing). Returns (loss, mse, tv_z, ssim, tv_xy, div,
        pc_loss, mse_ri, mse1, mse2).

        计算总损失及各个分量。Hxhat：渲染强度；xhat：重建的 RI 体；y：测量强度；
        steps：当前迭代步数（用于损失调度退火）。返回 (loss, mse, tv_z, ssim, tv_xy,
        div, pc_loss, mse_ri, mse1, mse2)。
        """
        # EN: Data-fidelity term. 'l12' blends L2 and L1: alpha decays linearly from 1 to 0
        # over decay_steps, so training starts L2-dominated and transitions to L1.
        # 中文：数据保真项。'l12' 混合 L2 与 L1：alpha 在 decay_steps 步内从 1 线性衰减到
        # 0，因此训练初期以 L2 为主，随后过渡到 L1。
        args.loss = 'l12'
        if args.loss == "l1":
            mse = torch.mean(torch.abs(Hxhat - y)) / 20
        elif args.loss == "l2":
            mse = torch.mean(torch.square(Hxhat - y)) / 20
        elif args.loss == "l12":
            decay_steps=500
            alpha=max((decay_steps-steps)/decay_steps,0)
            beta=max(steps/decay_steps,1)
            beta = min(beta, 1)
            mse1=torch.mean(torch.abs(Hxhat - y)) / 2
            mse2=torch.mean(torch.square(Hxhat - y)) / 2
            mse = alpha*mse2 + (1-alpha)*mse1
        else:
            raise NotImplementedError
        y.mean()
        Hxhat.mean()
        # 示例用法
        # perceptual_loss = PerceptualLoss()
        # pc_loss = perceptual_loss(Hxhat, y)
        pc_loss=0
        # regularizer
        # RI = np.load('/home/renzhihe/Desktop/phase_non_neural_real/RI_pred/RI.npy')
        # RI = torch.tensor(RI).cuda()
        # RI = RI.unsqueeze(0).permute(3, 0, 1, 2)
        # RI = RI / torch.max(RI)

        # RI_pred = self.DnCNN(xhat[:,:,:3].unsqueeze(0).permute(3, 0, 1, 2))

        # EN: Optional DnCNN denoiser-prior regularizer on the RI slices (disabled here by
        # forcing regularize_type to '').
        # 中文：可选的 DnCNN 去噪先验正则项，作用于 RI 切片（此处将 regularize_type 置空
        # 以禁用）。
        args.regularize_type = ''
        if args.regularize_type == "dncnn2d":
            # print(xhat.shape)
            # print(xhat.grad_fn)
            xhat_trans = torch.transpose(torch.squeeze(xhat), 3, 0)
            xhat_concat = torch.cat([xhat_trans[0, ...], xhat_trans[1, ...]], 2)
            xhat_concat = torch.transpose(xhat_concat, 2, 0)
            xhat_expand = xhat_concat.unsqueeze(1)
            with torch.no_grad():
                dncnn_loss = self.dncnn(xhat_expand)
            phase_regularize_value = (dncnn_loss.mean().squeeze()) * 1
            # phase_regularize_value = dncnn_loss(args, xhat_expand.to('cpu'), reuse=reuse)

            # 记得打开
            # phase_regularize_value = torch.tensor(0.0)
            absorption_regularize_value = torch.tensor(0.0)

        # EN: TV regularization on the RI volume in z and xy, with a weight that decays
        # linearly over the first decay_steps iterations (down to a small floor).
        # 中文：对 RI 体在 z 和 xy 方向施加 TV 正则，其权重在前 decay_steps 次迭代中
        # 线性衰减（保留一个较小的下限）。
        if 1:
            decay_steps = 200
            alpha_xy = max((decay_steps - steps) / decay_steps, 0.001)
            tv_z = alpha_xy * self.__total_variation_z(xhat)
            tv_xy = alpha_xy * tv_loss(xhat)

        # EN: SSIM structural term on the intensity images (converted to a dissimilarity:
        # (1 - SSIM) / 2, so 0 means identical images).
        # 中文：作用于强度图像的 SSIM 结构项（转换为差异度：(1 - SSIM) / 2，0 表示两图
        # 完全一致）。
        Hxhat = Hxhat.unsqueeze(1)
        y = y.unsqueeze(1)
        adaptive_ssim_ratio=1#+min(1*(steps)/3000,1)
        ssim = adaptive_ssim_ratio*(1 - self.SSIM(Hxhat, y)) / 2
        # print(ssim)


        # EN: Ground-truth supervised terms (RI MSE and diversity gap) — disabled by the
        # "and 0" guard; zeros are returned instead.
        # 中文：基于真值监督的项（RI 均方误差与多样性差异）——由 "and 0" 条件禁用，
        # 此时返回零值占位。
        if xhat_gt is not None and 0:
            mse_ri = torch.mean(torch.square(xhat - xhat_gt)) / 20

            div=diversity_loss(xhat)
            div_gt= diversity_loss(xhat_gt)
        else:
            mse_ri,div,div_gt=torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # print(steps)
        # EN: Fixed weights balancing the loss components; tv_z / tv_xy weights come from
        # the command-line args. The final loss sums the active terms (others commented out).
        # 中文：平衡各损失分量的固定权重；tv_z / tv_xy 的权重来自命令行参数。最终损失为
        # 各启用项之和（其余项已注释掉）。
        ratio_mse = 80
        ratio_ssim = 3
        ratio_tv_z = args.tv_z
        ratio_tv_xy = args.tv_xy
        ratio_div=0.4
        ratio_pc= 0.0001
        mse = mse * ratio_mse
        ssim = ssim * ratio_ssim
        tv_z = tv_z * ratio_tv_z
        tv_xy = tv_xy * ratio_tv_xy
        div=abs(div-div_gt)*ratio_div*1e3
        pc_loss = pc_loss * ratio_pc
        #tv_xy = tv_xy * ratio_tv_xy
        #phase_regularize_value = phase_regularize_value * ratio_reg
        #dark_xy = dark_xy*ratio_dark

        #final loss
        loss = (
                mse
                + ssim
                #+ (absorption_regularize_value + phase_regularize_value)
                + tv_z
                + tv_xy
                #+ pc_loss
                #+ div
                #+ dark_xy
        )

        return (
            loss,
            mse,
            tv_z,
            ssim,
            tv_xy,
            div,
            pc_loss,
            mse_ri,
            mse1,
            mse2
        )

    # EN: Anisotropic L1 total variation over the last two (spatial) dims of a 4D tensor.
    # 中文：对 4 维张量最后两个（空间）维度计算各向异性 L1 全变分。
    def __total_variation_2d(self, images):
        pixel_dif2 = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif3 = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        total_var = torch.sum(pixel_dif2) + torch.sum(pixel_dif3)
        return total_var

    # EN: Number of elements per sample (product of the non-batch dims), used for TV
    # normalization.
    # 中文：每个样本的元素个数（除批次维外各维之积），用于 TV 归一化。
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    # EN: Axial TV — L1 differences between adjacent z-slices of the RI volume.
    # 中文：轴向 TV——RI 体中相邻 z 层之间差值的 L1 范数。
    def __total_variation_z(self, images):
        """
        Normalized total variation 3d
        :param images: Images should have 4 dims: batch_size, z, x, y
        :return:
        """
        pixel_dif1 = torch.abs(images[:, :, 1:] - images[ :,:, :-1])
        total_var = torch.sum(pixel_dif1)
        return total_var
    # def __dncnn_inference(
    #     self,
    #     input,
    #     reuse,
    #     output_channel=1,
    #     layer_num=10,
    #     filter_size=3,
    #     feature_root=64,
    # ):
    #     # input layer
    #     with torch.no_grad():
    #         in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size//2)
    #         in_node = F.relu(in_node)
    #         # composite convolutional layers
    #         for layer in range(2, layer_num):
    #             in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size//2, bias=False)
    #             in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
    #         # output layer and residual learning
    #         in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size//2)
    #         output = input - in_node
    #     return output

    # def __dncnn_2d(self, args, images,reuse=True):  # [N, H, W, C]
    #     """
    #     DnCNN as 2.5 dimensional denoiser based on l-2 norm
    #     """
    #     a_min = args.DnCNN_normalization_min
    #     a_max = args.DnCNN_normalization_max
    #     normalized = (images - a_min) / (a_max - a_min)
    #     denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1),reuse)
    #     denormalized = denoised * (a_max - a_min) + a_min
    #     dncnn_res = torch.sum(denormalized**2)
    #     return dncnn_res


class dncnn_2d(nn.Module):
    """
    Lightweight DnCNN-style convolutional network used as a learned denoiser prior:
    an input conv, a stack of 3x3 convs with ReLU, and an output conv. Its forward pass
    normalizes the images, runs the denoiser, de-normalizes, and returns the squared
    L2 norm of the result as a regularization value. (Legacy component, unused by the
    default training loss.)

    轻量级的 DnCNN 风格卷积网络，用作学习到的去噪先验：由输入卷积层、若干带 ReLU
    的 3x3 卷积层和输出卷积层组成。前向过程先归一化图像，再经去噪网络处理并反归一
    化，最后以结果的 L2 平方范数作为正则值返回。（历史遗留组件，默认训练损失中未
    使用。）
    """
    def __init__(self, args, input_channel, output_channel=1, layer_num=10, filter_size=3, feature_root=64):
        super(dncnn_2d, self).__init__()
        self.input_conv = nn.Conv2d(input_channel, feature_root, filter_size, padding=filter_size // 2)
        self.convs = nn.ModuleList([
            nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False) for i in
            range(layer_num)
        ])
        self.output_conv = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)
        self.relu = nn.ReLU()

        # in_node = nn.Conv2d(input.size(1), feature_root, filter_size, padding=filter_size // 2)
        # in_node = F.relu(in_node)
        # # composite convolutional layers
        # for layer in range(2, layer_num):
        #     in_node = nn.Conv2d(feature_root, feature_root, filter_size, padding=filter_size // 2, bias=False)
        #     in_node = F.relu(nn.BatchNorm2d(feature_root)(in_node))
        # # output layer and residual learning
        # in_node = nn.Conv2d(feature_root, output_channel, filter_size, padding=filter_size // 2)

    def forward(self, args, images, reuse=True):
        a_min = args.DnCNN_normalization_min
        a_max = args.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(torch.clamp(normalized, 0, 1), reuse)
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = torch.sum(denormalized ** 2)
        return dncnn_res

        return 0

    # EN: Run the convolutional stack: input conv, hidden convs with ReLU, output conv.
    # 中文：执行卷积堆叠：输入卷积、带 ReLU 的隐藏卷积层、输出卷积。
    def __dncnn_inference(self, input, reuse=True):
        x = self.input_conv(input)
        for f in self.convs:
            x = f(x)
            x = self.relu(x)
        output = self.output_conv(x)

        return output