# Created by Renzhi He, COBI, UCDavis, 2023

"""
Trainable 3D refractive-index (RI) volume representations for FDT.
This module defines the NPRF model, a neural-field-style container that maps a
set of trainable parameters (an explicit voxel grid, an implicit MLP with
positional encoding, a tri-plane factorization, or per-voxel feature vectors)
to a 3D RI volume, and then renders intensity images through the multi-slice
beam-propagation forward model in optics.py. It also includes helper utilities
for checkpoint loading, data masking, and coarse-to-fine grid upsampling.

FDT 的可训练三维折射率（RI）体表示模块。
本模块定义了 NPRF 模型：一个类神经场（neural field）的容器，将一组可训练参数
（显式体素网格、带位置编码的隐式 MLP、三平面分解或逐体素特征向量）映射为三维
RI 体，并通过 optics.py 中的多层切片光束传播前向模型渲染出强度图像。
此外还包含用于加载权重、数据掩蔽以及由粗到细网格上采样的辅助工具函数。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
#import skimage
#from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
from absl import flags
import logging


from optics import PhaseObject3D, TomographySolver

#import contexttimer

FLAGS = flags.FLAGS

NUM_Z = "nz"
INPUT_CHANNEL = "ic"
OUTPUT_CHANNEL = "oc"
MODEL_SCOPE = "infer_y"
NET_SCOPE = "MLP"
DNCNN_SCOPE = "DnCNN"



# get total number of visible gpus
NUM_GPUS = torch.cuda.device_count()

# EN: Load a checkpoint into the model while tolerating different naming schemes
# (NeRF-style coarse/fine dicts, plain "network_state_dict", or DataParallel "module." prefixes).
# 中文：加载模型权重时兼容多种命名格式（NeRF 风格的 coarse/fine 字典、普通的
# "network_state_dict"，以及 DataParallel 的 "module." 前缀）。
def smart_load_state_dict(model: nn.Module, state_dict: dict):
    if "network_fn_state_dict" in state_dict.keys():
        state_dict_fn = {k.lstrip("module."): v for k, v in state_dict["network_fn_state_dict"].items()}
        state_dict_fn = {"mlp_coarse." + k: v for k, v in state_dict_fn.items()}

        state_dict_fine = {k.lstrip("module."): v for k, v in state_dict["network_fine_state_dict"].items()}
        state_dict_fine = {"mlp_fine." + k: v for k, v in state_dict_fine.items()}
        state_dict_fn.update(state_dict_fine)
        state_dict = state_dict_fn
    elif "network_state_dict" in state_dict.keys():
        state_dict = {k[7:]: v for k, v in state_dict["network_state_dict"].items()}
    else:
        state_dict = state_dict

    if isinstance(model, nn.DataParallel):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

# EN: Write a scalar value to a TensorBoard summary writer and flush immediately.
# 中文：向 TensorBoard 写入一个标量并立即刷新缓冲区。
def record_summary(writer, name, value, step):
    writer.add_scalar(name, value, step)
    writer.flush()


# EN: Reshape a 2D/3D image tensor to a 4D (batch, H, W, channel) layout expected downstream.
# 中文：将 2D/3D 图像张量整形为下游期望的 4 维（批次, 高, 宽, 通道）布局。
def reshape_image(image):
    if len(image.shape) == 2:
        image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    elif len(image.shape) == 3:
        image_reshaped = image.unsqueeze(-1)
    else:
        image_reshaped = image
    return image_reshaped

# EN: Randomly hold out a proportion of measurements along the first axis; returns the
# remaining data plus the removed/kept index lists (used for sparse-view experiments).
# 中文：沿第一维随机剔除一定比例的测量数据，返回剩余数据以及被剔除/保留的索引
# 列表（用于稀疏视角实验）。
def remove_data(array, proportion_to_remove):
    N = array.shape[0]
    num_to_remove = int(np.round(N * proportion_to_remove))
    shuffle_index=np.arange(N)
    np.random.shuffle(shuffle_index)
    indices_to_remove = shuffle_index[0:num_to_remove]
    indices_to_remove = np.sort(indices_to_remove)
    remaining_data = np.delete(array, indices_to_remove, axis=0)
    org_index=np.arange(N)
    indices_to_remain = np.delete(org_index, indices_to_remove, axis=0)
    return remaining_data, indices_to_remove,indices_to_remain

# EN: Inverse of remove_data: rebuild a full-size tensor by scattering the kept predictions
# back to their original indices and filling the held-out slots from the reference data.
# 中文：remove_data 的逆操作：将保留的预测值散射回原始索引位置，并用参考数据填充
# 被剔除的位置，从而重建完整尺寸的张量。
def insert_data_torch(original_tensor, data, indices_to_remove,indices_to_remain):
    new_tensor = torch.zeros(data.shape).cuda().float()
    new_tensor[indices_to_remain] = original_tensor.float()
    new_tensor[indices_to_remove] = torch.tensor(data[indices_to_remove]).float()
    return new_tensor

#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# EN: Custom autograd Function that bilinearly upsamples a parameter grid to a target
# resolution in the forward pass, and downsamples the incoming gradient back to the
# parameter's native resolution in the backward pass. This lets a low-resolution
# trainable grid drive a higher-resolution volume (coarse-to-fine training).
# 中文：自定义 autograd Function：前向时将参数网格双线性上采样到目标分辨率，
# 反向时把梯度插值回参数原始分辨率。由此可以用低分辨率的可训练网格驱动更高
# 分辨率的体（由粗到细训练）。
class InterpolateParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, target_size):
        ctx.shape = param.size()
        ctx.target_size = target_size
        # 使用插值，但保持param的形状不变
        return F.interpolate(param, size=(target_size, target_size), mode='bilinear', align_corners=False)

    @staticmethod
    def backward(ctx, grad_output):
        # 反插值到原始大小
        return F.interpolate(grad_output, size=ctx.shape[-2:], mode='bilinear', align_corners=False), None


class NPRF(nn.Module):
    """
    Neural physical representation of the refractive-index field, using an
    explicit trainable voxel grid (args.model = 'exp', optionally trained
    coarse-to-fine). The forward pass decodes the grid into an RI volume and
    renders intensity images through the multi-slice beam-propagation model.
    Also learns self-calibration terms: illumination source locations, voxel
    pitch (dx, dy, dz), and a per-image intensity gain.

    折射率场的神经物理表示，采用显式可训练体素网格（args.model = 'exp'，
    可选由粗到细训练）。前向过程将网格解码为 RI 体，并通过多层切片光束传播
    模型渲染强度图像。同时学习自校准量：照明光源位置、体素间距
    （dx、dy、dz）以及每张图像的强度增益。
    """
    def __init__(self, FLAGS,RI=None, locations=None, name="model_summary"):
        super(NPRF, self).__init__()
        args=FLAGS
        # EN: Optical and sampling parameters: wavelength, numerical aperture, background RI,
        # voxel pitch (dx/dy/dz scaled by the super-sampling factor) and grid dimensions.
        # 中文：光学与采样参数：波长、数值孔径、背景折射率、体素间距（dx/dy/dz 按超采样
        # 因子缩放）以及网格尺寸。
        # Setup parameters
        self.args = args
        self.name = name
        self.wavelength = args.wavelength
        self.NA = args.NA
        self.n_measure=args.n_measure
        self.n_b=args.n_b
        self.factor = args.factor

        self.dz = args.dz / self.factor
        self.layer = args.layers#int((layer+0.1) * 1)

        self.dx = args.dx / self.factor
        #
        grid_x = args.H / self.dx
        self.grid_x_org = int(grid_x * 1)
        self.grid_x = int(grid_x * 1)
        #self.dx = 2

        self.dy = args.dy / self.factor
        grid_y = args.W / self.dy #/ self.factor
        self.grid_y_org = int(grid_y * 1)
        self.grid_y = int(grid_y * 1)


        self.grid_x_org = args.grid_x
        self.grid_y_org = args.grid_y
        self.grid_x = args.grid_x
        self.grid_y = args.grid_y
        self.max_ri = args.max_ri
        if RI is None:
            self.refractive_update=np.random.rand(self.grid_x*self.grid_x*self.layer,1)*0.1
        # else:
        #     RI_pre=RI[(self.grid_x-self.shape_x)//2:-(self.grid_x-self.shape_x)//2, (self.grid_y-self.shape_y)//2:-(self.grid_y-self.shape_y)//2,1:]
        #     #RI_pre=
        #     self.refractive_update = RI_pre.reshape(self.shape_x*self.shape_y*self.layer,1)-self.n_b
        self.patch_ratio=args.patch_ratio

        ####selfcalibration
        # EN: Self-calibration parameters learned jointly with the volume: illumination
        # source locations and the physical voxel pitch (dx, dy, dz).
        # 中文：与体重建联合优化的自校准参数：照明光源位置以及物理体素间距
        # （dx、dy、dz）。
        self.locations=nn.Parameter(torch.tensor(locations))
        init_dx = getattr(args, "dx_init", self.dx)
        init_dy = getattr(args, "dy_init", self.dy)
        init_dz = getattr(args, "dz_init", self.dz)
        dxyz=torch.tensor([init_dx, init_dy, init_dz], dtype=torch.float32)
        self.dxyz=nn.Parameter(dxyz)
        # self.zz=nn.Parameter(torch.tensor([args.zz*100]))
        #print(self.locations.mean)
        #self.test=nn.Parameter(torch.tensor((1.)))
        
        # EN: Instantiate the volume parameterization: an explicit trainable voxel grid.
        # When coarse-to-fine (c2f) is enabled it starts at a low resolution and is
        # upsampled during training.
        # 中文：实例化体参数化方式：显式可训练体素网格。启用由粗到细（c2f）时先以
        # 低分辨率初始化，训练中逐步上采样。
        self.model=args.model
        if self.model=='exp':
            self.sampling=4
            if getattr(args, "c2f_enable", 0):
                init_block = int(args.c2f_stage_resolutions[0])
            else:
                init_block = int(self.grid_x)
            self.RI_init = nn.Parameter(torch.zeros((init_block, init_block, self.layer)))
        else:
            raise ValueError("model must be exp")

        # EN: Per-measurement intensity gain (one scalar per captured image) plus the
        # activation functions used throughout the model.
        # 中文：每个测量图像各一个标量的强度增益参数，以及模型中使用的各激活函数。
        self.per_img_gain = nn.Parameter(torch.ones(self.n_measure))
        self.le_relu = nn.LeakyReLU(negative_slope=args.relu_slope, inplace=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    ###########################
    ###     Neural Nets     ###
    ###########################
    def forward(self, light_loc_ids, training=True, steps=0, mask=None, ri_path=None, steps_c2f=None, block_sizes=None):
        """
        Full differentiable pipeline for one batch of illuminations: decode the trainable
        representation into an RI volume (or use a supplied volume when rendering), then
        simulate the intensity images via multi-slice beam propagation and apply per-image
        gains. Returns (RI offset or RI, intensity, dummy scalar, source locations).

        针对一批照明的完整可微流程：先将可训练表示解码为 RI 体（渲染模式下直接使用
        外部提供的体），再通过多层切片光束传播模拟强度图像并施加每图像增益。返回
        （RI 偏移量或 RI、强度图、占位标量、光源位置）。
        """
        # EN: Read the (learnable) voxel pitch from the self-calibration parameter.
        # 中文：从自校准参数中读取（可学习的）体素间距。
        self.dx = self.dxyz[0]
        self.dy = self.dxyz[1]
        self.dz = self.dxyz[2]
        # free_space=self.zz-self.dz*self.layer
        free_space = torch.tensor(self.args.fs)  # self.zz - self.dz * self.layer
        
        # Get light source locations
        light_loc = self.locations[light_loc_ids]
        
        # Choose RI computation method based on training mode and ri_path
        if not training and ri_path is not None:
            # EN: Render mode — bypass the trainable representation and propagate light
            # through an externally supplied RI volume (e.g. ground truth or a saved result).
            # 中文：渲染模式——跳过可训练表示，直接用外部提供的 RI 体（例如真值或已保存
            # 的重建结果）进行光传播。
            # Render mode: use provided ground truth RI
            print(f"Using provided RI for rendering, shape: {ri_path.shape}")
            RI = ri_path
            
            # # Ensure RI has the correct shape and device
            # if len(RI.shape) == 3:
            #     # RI shape should be (H, W, Z)
            #     if RI.shape != (self.grid_x, self.grid_y, self.layer):
            #         print(f"Resizing RI from {RI.shape} to ({self.grid_x}, {self.grid_y}, {self.layer})")
            #         # Resize RI to match expected dimensions
            #         RI = F.interpolate(
            #             RI.permute(2, 0, 1).unsqueeze(0), 
            #             size=(self.grid_x, self.grid_y), 
            #             mode='bilinear', 
            #             align_corners=False
            #         ).squeeze(0).permute(1, 2, 0)
                    
            #         # Adjust number of layers if needed
            #         if RI.shape[2] != self.layer:
            #             current_layers = RI.shape[2]
            #             if current_layers < self.layer:
            #                 # Pad with zeros
            #                 padding = torch.zeros(self.grid_x, self.grid_y, self.layer - current_layers, 
            #                                     dtype=RI.dtype, device=RI.device)
            #                 RI = torch.cat([RI, padding], dim=2)
            #             else:
            #                 # Truncate or interpolate
            #                 RI = F.interpolate(
            #                     RI.permute(2, 0, 1).unsqueeze(0),
            #                     size=(self.layer, self.grid_x, self.grid_y),
            #                     mode='trilinear',
            #                     align_corners=False
            #                 ).squeeze(0).permute(1, 2, 0)
            # else:
            #     raise ValueError(f"Expected RI to have 3 dimensions (H, W, Z), got {len(RI.shape)}")
            
            # Make sure RI is on the correct device
            if RI.device != next(self.parameters()).device:
                RI = RI.to(next(self.parameters()).device)
                
            # # Normalize RI values to expected range
            # # Assuming the input RI is in the range [0, 255] or [0, 1]
            # if RI.max() > 1.0:
            #     RI = RI / 255.0  # Normalize from [0, 255] to [0, 1]
            
            # # Convert to refractive index values
            # # Assuming we want RI values around 1.33 (water) + some variation
            # RI = RI * self.max_ri + self.n_b  # Scale and offset to proper RI range
            
            print(f"Final RI range: [{RI.min().item():.4f}, {RI.max().item():.4f}]")
            
        else:
            # Training mode: decode the trainable explicit grid into the RI volume
            # 中文：训练模式——将可训练的显式网格解码为 RI 体。
            if self.model == 'exp':
                # EN: Coarse-to-fine grid upsampling — at scheduled steps, bilinearly
                # resize the explicit grid parameter to the next block size and rewrap
                # it as a new nn.Parameter; afterwards always interpolate up to the full
                # working resolution (grid_x x grid_x).
                # 中文：由粗到细网格上采样——在预定的训练步数处，将显式网格参数双线性
                # 缩放到下一个块尺寸并重新封装为 nn.Parameter；随后总是插值到完整工作
                # 分辨率（grid_x x grid_x）。
                if steps_c2f is not None and steps in steps_c2f:
                    index = np.where(steps == np.array(steps_c2f))[0][0]
                    block_size = block_sizes[index]
                    self.block_size = block_size
                    if self.RI_init.shape[0] != block_size or self.RI_init.shape[1] != block_size:
                        RI = F.interpolate(
                            self.RI_init.permute(2, 0, 1).unsqueeze(0),
                            size=(block_size, block_size),
                            mode='bilinear',
                        ).squeeze(0).permute(1, 2, 0)
                        self.RI_init = nn.Parameter(RI, requires_grad=True)
                RI = F.interpolate(self.RI_init[:, :, :].permute(2, 0, 1).unsqueeze(0),
                                size=(self.grid_x, self.grid_x), mode='bilinear').squeeze(0).permute(1, 2, 0)
            else:
                raise ValueError("model must be exp")
            
            # Apply neural network transformations (only in training mode)
            # EN: Map raw outputs to physical RI: sigmoid squashes to (-max_ri, max_ri),
            # LeakyReLU suppresses negative contrast, then the background RI n_b is added.
            # 中文：将原始输出映射为物理 RI：sigmoid 压缩到 (-max_ri, max_ri)，LeakyReLU
            # 抑制负的折射率对比度，最后加上背景折射率 n_b。
            # last_layer=torch.zeros(1024,1024,1)
            # RI = torch.cat((RI, last_layer), dim=2)
            RI = (self.sigmoid(RI) - 0.5) / 0.5 * self.max_ri  # -0.1~0.1
            RI = self.le_relu(RI)  # 0-0.1
            RI = RI + self.n_b

        # print(torch.mean(RI))
        # patch=torch.ones((RI.shape[0],RI.shape[1],1))*1.33
        # RI=torch.cat((RI,patch),dim=2)
        
        # Render intensity using the computed or provided RI
        # EN: Physics-based rendering: propagate each illumination through the RI volume.
        # 中文：基于物理的渲染：让每个照明光源穿过 RI 体进行传播。
        intensity = self.rendering(light_loc, RI, free_space)
        
        # ---------- 应用 per-image gain（关键新增） ----------
        # per_img_gain 的形状是 [n_measure]，在这里根据 light_loc_ids_t 取出当前 batch 的 gain
        gain = self.per_img_gain[light_loc_ids]                  # [B]
        gain = gain.view(-1, 1, 1)                               # [B, 1, 1]
        intensity = intensity * gain                             # [B, H, W]

        # Return results
        # In render mode, we return the actual RI values rather than RI-1.33
        dummy = torch.tensor(0.0, device=intensity.device)
        if not training:
            return RI, intensity, dummy, self.locations
        else:
            return RI - 1.33, intensity, dummy, self.locations


    def rendering(self, light_source, refractive_index,free_space=75):
        """
        Convert light-source grid positions into physical illumination spatial frequencies
        (fx, fy) and run the multi-slice forward model to obtain the predicted intensity
        images for this batch of sources.

        将光源网格坐标换算为物理照明空间频率（fx、fy），并运行多层切片前向模型，得到
        该批光源对应的预测强度图像。
        """
        # print("refractive index shape:", refractive_index.shape)
        # print("light source shape:", light_source.shape)
        # print("input:",self.input[0])
        self.wavelength = 0.6  # fluorescence wavelength
        # objective immersion media

        # background refractive index, PDMS
        self.n_b = 1.33
        #fx_illu_list = (light_source[ :, 0] - self.grid_x//2)*self.dx/self.grid_x
        #fy_illu_list = (light_source[ :, 1] - self.grid_y // 2) * self.dy / self.grid_y
        fx_illu_list = light_source[ :, 0]*self.dx
        fy_illu_list = light_source[ :, 1]*self.dy
        fz_illu_list = torch.zeros_like(fx_illu_list)#light_source[:,2]
        intensityfield=self.multislice(refractive_index,  fx_illu_list=fx_illu_list, fy_illu_list=fy_illu_list, fz_illu_list=fz_illu_list, dx=self.dx,
                   dy=self.dy, dz=self.dz,free_space=free_space)

        return intensityfield

    def multislice(self, refractive_index, fx_illu_list, fy_illu_list, fz_illu_list, dx=0.2, dy=0.2, dz=0.2,free_space=75):
        """
        Wrap the RI volume in a PhaseObject3D, configure a TomographySolver with the
        MultiPhaseContrast (multi-slice beam propagation) scattering model, run the forward
        prediction, and return the intensity |E|^2 of the propagated field.

        将 RI 体封装为 PhaseObject3D，用 MultiPhaseContrast（多层切片光束传播）散射模型
        配置 TomographySolver，执行前向预测，并返回传播光场的强度 |E|^2。
        """
        # Setup solver objects
        solver_params = dict(wavelength=self.wavelength, na=self.NA, \
                             RI_measure=self.n_measure, sigma=2 * np.pi * dz / self.wavelength, \
                             fx_illu_list=fx_illu_list, fy_illu_list=fy_illu_list, fz_illu_list=fz_illu_list, \
                             pad=False, pad_size=(50, 50))
        ## add value to the phantom
        phase_obj_3d = PhaseObject3D(shape=refractive_index.shape, RI_obj=refractive_index,voxel_size=(dy, dx, dz), RI=self.n_b,free_space=free_space,args=self.args)
        #phase_obj_3d.RI_obj[grid_x//2-50:grid_x//2+50, grid_y//2-50:grid_y//2+50,:] = phase_obj_3d.RI_obj[grid_x//2-50:grid_x//2+50, grid_y//2-50:grid_y//2+50,:] + refractive_index
        #phase_obj_3d.RI_obj=refractive_index
        solver_obj = TomographySolver(phase_obj_3d, **solver_params)
        solver_obj.setScatteringMethod(model="MultiPhaseContrast")
        forward_field_mb, fields = solver_obj.forwardPredict(field=True)

        forward_field_mb = torch.squeeze(torch.stack(forward_field_mb))
        intensityfield = torch.abs(forward_field_mb * torch.conj(forward_field_mb))
        return intensityfield
    # EN: Save the model state_dict (and optionally the data provider) into an
    # epoch-numbered or "latest" subdirectory.
    # 中文：将模型 state_dict（以及可选的数据提供器）保存到按 epoch 编号或名为
    # "latest" 的子目录中。
    def save(self, directory, epoch=None, train_provider=None):
        if epoch is not None:
            directory = os.path.join(directory, "{}_model/".format(epoch))
        else:
            directory = os.path.join(directory, "latest/".format(epoch))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "model")
        if train_provider is not None:
            train_provider.save(directory)
        torch.save(self.state_dict(), path)
        print("saved to {}".format(path))
        return path

    # EN: Restore weights from a checkpoint file (non-strict, so partial matches load).
    # 中文：从检查点文件恢复权重（非严格模式，允许部分参数匹配加载）。
    def restore(self, model_path):

        param = torch.load(model_path)
        # param_model=self.state_dict()
        # new_dict={}
        # for k,v in param.items():
        #     if k in param_model:
        #         print(k)
        #         print(v)
        self.load_state_dict(param, strict=False)


    # EN: Directly overwrite the stored RI update buffer with an external volume.
    # 中文：用外部提供的体数据直接覆盖内部保存的 RI 更新缓存。
    def load_ri(self,RI):
        self.refractive_update=RI





