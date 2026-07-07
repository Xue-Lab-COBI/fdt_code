"""
EN: Structural Similarity Index (SSIM) metric implemented in PyTorch. SSIM compares
two images through local statistics (mean, variance, covariance) computed with a
Gaussian window, and is used in FDT to evaluate how well rendered intensity images
match the measured ones. Provides both a functional interface (ssim) and a reusable
nn.Module (SSIM) that caches the Gaussian window across calls.

中文：基于 PyTorch 实现的结构相似性指标（SSIM）。SSIM 通过高斯窗口计算的局部统计量
（均值、方差、协方差）来比较两幅图像，在 FDT 中用于评估前向模型渲染的强度图像与实测
图像的一致程度。本模块同时提供函数式接口（ssim）和可复用的 nn.Module（SSIM 类），
后者会在多次调用之间缓存高斯窗口以避免重复构建。
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

# EN: Build a normalized 1D Gaussian kernel of the given size and standard deviation.
# 中文：生成给定尺寸和标准差的一维高斯核，并归一化使其权重之和为 1。
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

# EN: Create a 2D Gaussian window (outer product of two 1D kernels) expanded to the
# requested channel count, shaped for use as a depthwise conv2d filter.
# 中文：通过两个一维高斯核的外积构造二维高斯窗口，并扩展到指定通道数，
# 形状适配于按通道分组的 conv2d 卷积核。
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# EN: Core SSIM computation. Local means/variances/covariance are estimated by Gaussian
# filtering, then combined with the standard stability constants C1/C2 into the SSIM map;
# the result is averaged over the whole batch or per image depending on size_average.
# 中文：SSIM 核心计算。先用高斯滤波估计局部均值、方差和协方差，再结合标准稳定常数
# C1/C2 得到 SSIM 图；根据 size_average 对整个 batch 取平均或逐图取平均。
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# EN: Module wrapper around _ssim that caches the Gaussian window; it is rebuilt only
# when the input channel count or dtype/device changes (see forward).
# 中文：对 _ssim 的模块化封装，缓存高斯窗口；仅当输入的通道数或 dtype/设备
# 发生变化时才重新构建（见 forward）。
class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
    def forward(self, img1, img2):
        # print(img1.shape)
        # print(img2.shape)
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# EN: Functional one-shot SSIM: builds a fresh window matched to the input's channels
# and device, then delegates to _ssim.
# 中文：函数式一次性 SSIM 接口：根据输入的通道数和设备即时构建高斯窗口，
# 然后调用 _ssim 完成计算。
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
