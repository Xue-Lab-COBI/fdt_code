"""
EN: Main training and rendering entry point for FDT (refractive-index tomography).
This script reconstructs a 3D refractive-index (RI) volume from measured
intensity images: a multi-slice beam-propagation forward model (optics.py)
simulates image formation, while a neural-field-style trainable volume
(NeRF.py, class NPRF) represents the sample. It supports optional
self-calibration of the illumination source locations and of the voxel
pitch (dx/dy/dz), coarse-to-fine training schedules, checkpointing, and a
final quantitative evaluation.

中文：FDT（折射率断层成像）的主训练/渲染入口脚本。
本脚本从测量得到的强度图像重建三维折射率（RI）体数据：使用多层切片的
光束传播正向模型（optics.py）模拟成像过程，并用类神经场的可训练体
（NeRF.py 中的 NPRF 类）表示样品。支持照明光源位置及体素间距
（dx/dy/dz）的自校准、由粗到细（coarse-to-fine）的训练策略、
断点保存与最终的定量评估。
"""
# Created by Renzhi He, COBI, UCDavis, 2023
# 11/22 test self calibration
import os
import time
import glob

import cv2
import torch
import torch.nn as nn
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from NeRF import *
from load_data import *
# from run_nerf_helpers import *
#from metrics import compute_img_metric
from loss import Loss
# np.random.seed(0)
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path

# EN: Optional dependency: tifffile for saving TIFF volumes.
# 中文：可选依赖：tifffile 用于保存 TIFF 体数据。
try:
    import tifffile
except ImportError:
    tifffile = None
import itertools

# EN: Utility that writes an mp4 video where each frame shows the normalized image with min/max/mean statistics overlaid.
# 中文：工具函数：生成 mp4 视频，每帧叠加显示归一化图像及其最小/最大/平均值统计信息。
def create_video_with_stats(imgs, video_path, fps=10):
    """
    创建视频，每帧显示归一化图像和统计信息
    
    参数:
    - imgs: 图像数组，形状为 (N, H, W)
    - video_path: 输出视频路径
    - fps: 帧率
    """
    height, width = imgs.shape[1], imgs.shape[2]
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    print(f"Creating video with {imgs.shape[0]} frames...")
    
    for i in range(imgs.shape[0]):
        img = imgs[i]
        
        # 计算统计信息
        img_min = np.min(img)
        img_max = np.max(img)
        img_mean = np.mean(img)
        
        # 归一化图像到0-255
        if img_max > img_min:
            img_norm = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img, dtype=np.uint8)
        
        # 转换为3通道图像以便添加彩色文字
        img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        
        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)  # 绿色
        thickness = 2
        
        # 文字位置
        y_offset = 30
        
        # 添加帧号
        cv2.putText(img_color, f'Frame: {i:04d}', (10, y_offset), 
                   font, font_scale, color, thickness)
        
        # 添加最小值
        cv2.putText(img_color, f'Min: {img_min:.4f}', (10, y_offset + 30), 
                   font, font_scale, color, thickness)
        
        # 添加最大值
        cv2.putText(img_color, f'Max: {img_max:.4f}', (10, y_offset + 60), 
                   font, font_scale, color, thickness)
        
        # 添加平均值
        cv2.putText(img_color, f'Mean: {img_mean:.4f}', (10, y_offset + 90), 
                   font, font_scale, color, thickness)
        
        # 写入视频帧
        video_writer.write(img_color)
        
        # 显示进度
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{imgs.shape[0]} frames")
    # 释放视频写入器
    video_writer.release()
    print(f"Video saved to: {video_path}")

def apply_custom_colormap(gray_image, palette):
    """
    EN: Map a grayscale image to a color image using a lookup-table palette
    (one RGB triplet per gray level after min-max normalization).

    中文：使用查找表调色板将灰度图映射为彩色图（先做最小-最大归一化，
    每个灰度级对应一个 RGB 颜色）。
    """
    # Normalize the grayscale image to have values between 0 and len(palette)-1
    normalized_gray = cv2.normalize(gray_image, None, 0, len(palette) - 1, cv2.NORM_MINMAX)

    # Create an empty color image
    colored_image = np.zeros((*gray_image.shape, 3), dtype=np.uint8)

    # Apply the palette
    for i in range(len(palette)):
        colored_image[normalized_gray == i] = palette[i]

    return colored_image
# EN: Load a custom colormap from disk if available; otherwise fall back to a plain grayscale ramp.
# 中文：如磁盘上存在自定义调色板文件则加载，否则退回到简单的灰度渐变。
if os.path.exists("colormap0627.npy"):
    palette = np.load("colormap0627.npy")
    palette = palette[:, [2, 1, 0]]
else:
    palette = np.stack(
        [
            np.arange(256, dtype=np.uint8),
            np.arange(256, dtype=np.uint8),
            np.arange(256, dtype=np.uint8),
        ],
        axis=1,
    )


def _bool_flag(value):
    """
    EN: Convert an int-like flag ("0"/"1", 0/1) to a Python bool.

    中文：将整型风格的开关值（"0"/"1"、0/1）转换为布尔值。
    """
    return bool(int(value))


def _format_tag(value):
    """
    EN: Format a numeric value into a compact tag for experiment names
    (integers keep their form, decimals drop the dot).

    中文：将数值格式化为实验名中使用的紧凑标签（整数保持原样，小数去掉小数点）。
    """
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "")


def build_experiment_name(args):
    """
    EN: Build a descriptive experiment name that encodes the key settings
    (dataset, feature size, layer count, max RI, noise/self-calibration flags,
    coarse-to-fine flag, background RI, etc.) so runs are self-documenting.

    中文：根据关键配置（数据集、特征尺寸、层数、最大折射率、噪声/自校准开关、
    由粗到细开关、背景折射率等）构造带说明性的实验名称，使每次运行可自描述。
    """
    ri_tag = _format_tag(round(args.max_ri, 3))
    return (
        f"{args.data_name}_fs{_format_tag(args.fs)}_layer{args.layers}_ri{ri_tag}"
        f"_ln{int(_bool_flag(args.location_noise_enable))}"
        f"_dn{int(_bool_flag(getattr(args, 'dxyz_noise_enable', 0)))}"
        f"_scp{int(_bool_flag(getattr(args, 'position_calibration_enable', 0)))}"
        f"_scd{int(_bool_flag(getattr(args, 'dxyz_calibration_enable', 0)))}"
        f"_c2f{int(_bool_flag(args.c2f_enable))}"
        f"_b2c{args.b2c}"
        f"_b2b{args.b2b}"
        f"_NB{_format_tag(args.n_b)}"
    )


def get_stage_schedule(args):
    """
    EN: Apply the training policy and return the coarse-to-fine schedule as
    (stage start steps, stage grid resolutions). When coarse-to-fine is
    disabled, a single full-resolution stage is used.

    中文：先应用训练策略，然后返回由粗到细的训练计划（每个阶段的起始步数
    与对应的网格分辨率）。若未启用由粗到细，则只使用单个全分辨率阶段。
    """
    apply_training_policy(args)

    if not _bool_flag(args.c2f_enable):
        return [0], [args.grid_x]

    stage_steps = [int(step) for step in args.c2f_stage_steps]
    stage_resolutions = [int(res) for res in args.c2f_stage_resolutions]
    if len(stage_steps) != len(stage_resolutions):
        raise ValueError("c2f_stage_steps and c2f_stage_resolutions must have the same length")
    if not stage_steps or stage_steps[0] != 0:
        raise ValueError("c2f_stage_steps must start with 0")
    if stage_steps != sorted(stage_steps):
        raise ValueError("c2f_stage_steps must be sorted in ascending order")
    return stage_steps, stage_resolutions


def get_stage_resolution(step, stage_steps, stage_resolutions):
    """
    EN: Return the grid resolution of the coarse-to-fine stage that the given
    global step falls into.

    中文：根据当前全局步数，返回其所处由粗到细阶段对应的网格分辨率。
    """
    stage_idx = 0
    for idx, stage_start in enumerate(stage_steps):
        if step >= stage_start:
            stage_idx = idx
        else:
            break
    return stage_resolutions[stage_idx]


def build_location_initializer(light_loc_gt, args):
    """
    EN: Build the initial illumination source locations. If location noise is
    enabled, add scaled Gaussian noise to the ground-truth locations (z is
    reset to 0) so the self-calibration has a perturbed starting point.

    中文：构造照明光源位置的初值。若启用位置噪声，则在真值位置上加入按幅度
    缩放的高斯噪声（z 分量重置为 0），从而为自校准提供一个带扰动的起点。
    """
    if not _bool_flag(args.location_noise_enable):
        print("No adding location noise")
        return light_loc_gt

    rng = np.random.default_rng(args.location_noise_seed)
    noise = rng.normal(0.0, args.location_noise_std, light_loc_gt.shape).astype(np.float32)
    max_abs = np.max(np.abs(noise))
    if max_abs > 0 and args.location_noise_scale > 0:
        noise = noise / max_abs * args.location_noise_scale
    else:
        noise[:] = 0.0

    light_loc = light_loc_gt + noise
    light_loc[:, 2] = 0
    print(
        "Added location noise",
        f"std={args.location_noise_std}",
        f"scale={args.location_noise_scale}",
        f"seed={args.location_noise_seed}",
    )
    return light_loc


def build_dxyz_initializer(args):
    """
    EN: Build the initial voxel pitch (dx, dy, dz). If dxyz noise is enabled,
    apply multiplicative Gaussian perturbations (clamped to at least 5% of the
    ground-truth pitch) to test pitch self-calibration.

    中文：构造体素间距 (dx, dy, dz) 的初值。若启用 dxyz 噪声，则施加乘性
    高斯扰动（下限约束为真值间距的 5%），用于测试体素间距的自校准能力。
    """
    dxyz_gt = np.array([args.dx, args.dy, args.dz], dtype=np.float32)
    if not _bool_flag(getattr(args, "dxyz_noise_enable", 0)):
        print("No adding dxyz noise")
        return dxyz_gt

    rng = np.random.default_rng(args.dxyz_noise_seed)
    noise = rng.normal(0.0, args.dxyz_noise_std, dxyz_gt.shape).astype(np.float32)
    max_abs = np.max(np.abs(noise))
    if max_abs > 0 and args.dxyz_noise_scale > 0:
        noise = noise / max_abs * args.dxyz_noise_scale
    else:
        noise[:] = 0.0

    dxyz = dxyz_gt * (1.0 + noise)
    dxyz = np.maximum(dxyz, dxyz_gt * 0.05)
    print(
        "Added dxyz noise",
        f"std={args.dxyz_noise_std}",
        f"scale={args.dxyz_noise_scale}",
        f"seed={args.dxyz_noise_seed}",
        f"init={dxyz.tolist()}",
    )
    return dxyz


def build_optimizer(args, nerf):
    """
    EN: Create the Adam optimizer with two parameter groups: the main model
    parameters (volume, dxyz) at args.lrate, and the source-location
    parameters at args.position_lrate.

    中文：创建 Adam 优化器，包含两个参数组：主模型参数（体数据、dxyz）
    使用 args.lrate，光源位置参数使用 args.position_lrate。
    """
    # EN: Split source-location parameters from the rest so they get their own learning rate.
    # 中文：将光源位置参数与其余参数分开，使其使用独立的学习率。
    location_params = []
    main_params = []
    for name, param in nerf.named_parameters():
        if name.endswith("locations"):
            if not param.requires_grad and not _bool_flag(getattr(args, "position_calibration_enable", 0)):
                continue
            location_params.append(param)
        else:
            if not param.requires_grad:
                if name.endswith("dxyz") and _bool_flag(getattr(args, "dxyz_calibration_enable", 0)):
                    main_params.append(param)
                    continue
                continue
            main_params.append(param)

    # EN: Assemble per-group learning rates; group names are used later for staged LR updates.
    # 中文：为各参数组设置学习率；组名在之后的分阶段学习率更新中使用。
    param_groups = []
    if main_params:
        param_groups.append(
            {
                "params": main_params,
                "lr": args.lrate,
                "group_name": "main",
            }
        )
    if location_params:
        param_groups.append(
            {
                "params": location_params,
                "lr": args.position_lrate,
                "group_name": "locations",
            }
        )

    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))


def set_calibration_trainability(nerf, args, global_step):
    """
    EN: Enable/disable gradients on the source locations and voxel pitch
    according to the calibration flags and the delayed start step
    (self_calibration_step). Returns whether calibration is currently active.

    中文：根据自校准开关及延迟启动步数（self_calibration_step），
    打开或关闭光源位置和体素间距参数的梯度。返回当前自校准是否已激活。
    """
    calibration_active = global_step >= int(getattr(args, "self_calibration_step", 0))
    position_active = _bool_flag(getattr(args, "position_calibration_enable", 0)) and calibration_active
    dxyz_active = _bool_flag(getattr(args, "dxyz_calibration_enable", 0)) and calibration_active

    nerf.module.locations.requires_grad_(position_active)
    nerf.module.dxyz.requires_grad_(dxyz_active)
    return calibration_active


def apply_training_policy(args):
    """
    EN: Normalize the self-calibration flags: the coarse switch
    self_calibration_enable turns on both position and dxyz calibration, and
    is then recomputed as the OR of the two fine-grained flags.

    中文：统一整理自校准相关开关：总开关 self_calibration_enable 会同时
    启用位置校准与 dxyz 校准；随后它被重新计算为两个细粒度开关的逻辑或。
    """
    if _bool_flag(args.self_calibration_enable) and not (
        _bool_flag(getattr(args, "position_calibration_enable", 0))
        or _bool_flag(getattr(args, "dxyz_calibration_enable", 0))
    ):
        args.position_calibration_enable = 1
        args.dxyz_calibration_enable = 1

    args.self_calibration_enable = int(
        _bool_flag(getattr(args, "position_calibration_enable", 0))
        or _bool_flag(getattr(args, "dxyz_calibration_enable", 0))
    )


def get_lr_schedule(args):
    """
    EN: Return the piecewise-constant learning-rate schedule (boundary steps
    and stage values) and reset args.lrate to the first stage value.

    中文：返回分段常数的学习率计划（阶段边界步数与各阶段学习率），
    并将 args.lrate 重置为第一阶段的学习率。
    """
    stage_steps = list(getattr(args, "lr_stage_steps", [150, 500, 750]))
    stage_values = list(getattr(args, "lr_stage_values", [args.lrate]))
    if stage_values:
        args.lrate = float(stage_values[0])
    return stage_steps, stage_values


def get_main_learning_rate(args, global_step):
    """
    EN: Look up the main-group learning rate for the current global step
    from the piecewise-constant schedule.

    中文：根据分段常数学习率计划，查询当前全局步数对应的主参数组学习率。
    """
    stage_steps, stage_values = get_lr_schedule(args)
    if not stage_values:
        return float(args.lrate)

    for idx, step in enumerate(stage_steps):
        if global_step < step:
            return float(stage_values[min(idx, len(stage_values) - 1)])
    return float(stage_values[-1])


def normalize_pred_volume_for_metrics(volume):
    """
    EN: Prepare a predicted RI volume for metric computation: clip negatives
    to zero and scale by the maximum so values lie in [0, 1].

    中文：为指标计算预处理预测的 RI 体数据：负值截断为 0，
    再除以最大值使数值落入 [0, 1]。
    """
    volume = volume.astype(np.float32).copy()
    volume[volume < 0] = 0
    vmax = np.max(volume)
    if vmax > 0:
        volume = volume / vmax
    return volume


def normalize_gt_volume_for_metrics(volume):
    """
    EN: Min-max normalize the ground-truth RI volume to [0, 1] for metric
    computation (constant volumes map to all zeros).

    中文：为指标计算将真值 RI 体数据做最小-最大归一化到 [0, 1]
    （常数体数据将映射为全零）。
    """
    volume = volume.astype(np.float32).copy()
    vmin = np.min(volume)
    vmax = np.max(volume)
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume)
    volume[volume < 0] = 0
    return volume


def evaluate_ri_metrics(ri_pred, ri_gt):
    """
    EN: Compute slice-wise volume metrics (MSE / SSIM / LPIPS / PSNR / PCC /
    perceptual loss) between the normalized predicted and ground-truth RI
    volumes, returning a dictionary of scalars.

    中文：在归一化后的预测与真值 RI 体数据之间按切片计算体指标
    （MSE / SSIM / LPIPS / PSNR / PCC / 感知损失），返回标量字典。
    """
    ri_pred = normalize_pred_volume_for_metrics(ri_pred)
    ri_gt = normalize_gt_volume_for_metrics(ri_gt)
    pred_t = torch.tensor(ri_pred).permute(2, 0, 1).unsqueeze(1).float()
    gt_t = torch.tensor(ri_gt).permute(2, 0, 1).unsqueeze(1).float()
    mse, ssim, lpi, psnr, pcc, pc = metrics()(pred_t, gt_t)
    return {
        "mse": float(mse.item()),
        "ssim": float(ssim.item()),
        "lpips": float(lpi.mean().item()),
        "psnr": float(psnr.item()),
        "pcc": float(pcc.item()),
        "pc": float(pc.item()),
    }


def evaluate_image_metrics(snapshot_dir):
    """
    EN: Load matching predicted/ground-truth image pairs saved in a snapshot
    directory and compute the same set of image metrics; returns None when no
    valid pairs are found.

    中文：从快照目录读取成对的预测/真值图像并计算同一组图像指标；
    若没有有效图像对则返回 None。
    """
    pred_paths = sorted(glob.glob(os.path.join(snapshot_dir, "img_*_pred.png")))
    pairs = []
    for pred_path in pred_paths:
        gt_path = pred_path.replace("_pred.png", "_gt.png")
        if os.path.exists(gt_path):
            pairs.append((pred_path, gt_path))
    if not pairs:
        return None

    pred_imgs = []
    gt_imgs = []
    for pred_path, gt_path in pairs:
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        pred_imgs.append(pred.astype(np.float32) / 255.0)
        gt_imgs.append(gt.astype(np.float32) / 255.0)
    if not pred_imgs:
        return None

    pred_t = torch.tensor(np.stack(pred_imgs)).unsqueeze(1).float()
    gt_t = torch.tensor(np.stack(gt_imgs)).unsqueeze(1).float()
    mse, ssim, lpi, psnr, pcc, pc = metrics()(pred_t, gt_t)
    return {
        "count": len(pred_imgs),
        "mse": float(mse.item()),
        "ssim": float(ssim.item()),
        "lpips": float(lpi.mean().item()),
        "psnr": float(psnr.item()),
        "pcc": float(pcc.item()),
        "pc": float(pc.item()),
    }


def find_latest_snapshot_dir(exp_dir):
    """
    EN: Find the numerically largest step-numbered snapshot subdirectory under
    the experiment directory, or None if none exist.

    中文：在实验目录下查找以步数命名的快照子目录中数值最大的一个；
    若不存在则返回 None。
    """
    step_dirs = [
        os.path.join(exp_dir, name)
        for name in os.listdir(exp_dir)
        if name.isdigit() and os.path.isdir(os.path.join(exp_dir, name))
    ]
    if not step_dirs:
        return None
    return max(step_dirs, key=lambda path: int(os.path.basename(path)))


def run_final_evaluation(exp_dir, data_path):
    """
    EN: After training, evaluate the latest snapshot: RI-volume metrics against
    RI_gt.npy (if available) and image metrics over saved pred/gt pairs, then
    write the summary to final_evaluation.txt in the experiment directory.

    中文：训练结束后对最新快照进行评估：若存在 RI_gt.npy 则计算 RI 体指标，
    并对保存的预测/真值图像对计算图像指标，最后将结果写入实验目录下的
    final_evaluation.txt。
    """
    latest_dir = find_latest_snapshot_dir(exp_dir)
    if latest_dir is None:
        print("Skip final evaluation: no snapshot directory found")
        return

    lines = [f"latest_dir={latest_dir}"]
    ri_path = os.path.join(latest_dir, "RI.npy")
    ri_gt_path = os.path.join(data_path, "RI_gt.npy")
    if os.path.exists(ri_path) and os.path.exists(ri_gt_path):
        ri_metrics = evaluate_ri_metrics(np.load(ri_path), np.load(ri_gt_path))
        lines.append(
            "RI: "
            + ",".join(
                [
                    f"mse:{ri_metrics['mse']:.4e}",
                    f"ssim:{ri_metrics['ssim']:.4f}",
                    f"lpips:{ri_metrics['lpips']:.4f}",
                    f"psnr:{ri_metrics['psnr']:.4f}",
                    f"pcc:{ri_metrics['pcc']:.4f}",
                    f"pc:{ri_metrics['pc']:.4f}",
                ]
            )
        )
    else:
        lines.append("RI: skipped")

    image_metrics = evaluate_image_metrics(latest_dir)
    if image_metrics is not None:
        lines.append(
            "IMAGE: "
            + ",".join(
                [
                    f"count:{image_metrics['count']}",
                    f"mse:{image_metrics['mse']:.4e}",
                    f"ssim:{image_metrics['ssim']:.4f}",
                    f"lpips:{image_metrics['lpips']:.4f}",
                    f"psnr:{image_metrics['psnr']:.4f}",
                    f"pcc:{image_metrics['pcc']:.4f}",
                    f"pc:{image_metrics['pc']:.4f}",
                ]
            )
        )
    else:
        lines.append("IMAGE: skipped")

    eval_path = os.path.join(exp_dir, "final_evaluation.txt")
    with open(eval_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Final evaluation saved to {eval_path}")


# EN: The helpers below centralize the output directory and file-path conventions of an experiment.
# 中文：以下辅助函数统一约定单次实验的输出目录与文件路径。
def get_experiment_dir(args, expname=None):
    if expname is None:
        expname = args.object_category_ori
    return os.path.join(args.basedir, expname)


def get_tensorboard_dir(exp_dir):
    return os.path.join(exp_dir, "tensorboard")


def get_preview_dir(exp_dir):
    return os.path.join(exp_dir, "image_pred")


def get_ri_dir(exp_dir):
    return os.path.join(exp_dir, "RI_pred")


def get_summary_metrics_path(exp_dir):
    return os.path.join(exp_dir, "metricsRI.txt")

class PerceptualLoss(nn.Module):
    """
    EN: VGG16-feature-based perceptual loss. If torchvision/VGG weights are
    unavailable the loss silently degrades to zero.

    中文：基于 VGG16 特征的感知损失。若 torchvision 或 VGG 权重不可用，
    损失将静默退化为 0。
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = None
        try:
            from torchvision.models import vgg16

            vgg = vgg16(pretrained=True).features[:10]
            self.vgg = vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        except Exception:
            self.vgg = None

    def forward(self, input, target):
        if self.vgg is None:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)#.transpose(1,0)#[20]
            target = target.repeat(1, 3, 1, 1)#.transpose(1,0)#[20]
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = nn.functional.mse_loss(input_features, target_features)
        return loss

class metrics(nn.Module):
    """
    EN: Bundle of image-quality metrics (MSE, SSIM, LPIPS, PSNR, Pearson
    correlation, perceptual loss) used to evaluate reconstructed volumes and
    rendered images against ground truth.

    中文：图像质量指标集合（MSE、SSIM、LPIPS、PSNR、皮尔逊相关系数、
    感知损失），用于将重建体数据与渲染图像同真值进行对比评估。
    """
    def __init__(self, DnCNNN_channels=1, tower_idx=None, Hreal=None, Himag=None):
        super(metrics, self).__init__()
        self.tower_idx = tower_idx
        self.Hreal = Hreal
        self.Himag = Himag
        from ssim import SSIM
        self.SSIM = SSIM()
        self.pc=PerceptualLoss()
        self.lpips_fn = None
        self.lpips_import_error = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').eval()
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        except Exception as exc:
            self.lpips_import_error = exc


    ##############################
    ###     Loss Functions     ###
    ##############################
    def calculate_lpips(self,img1, img2):
        """
        Calculate the LPIPS metric between two images.

        Parameters:
        - img1, img2: tensors representing the two images to compare.
                      They should have a shape of (B, C, H, W) and be normalized to the range [-1, 1].
        - net_type: the type of pretrained network to use ('alex', 'vgg', etc.). 'alex' is commonly used.
        - use_gpu: a boolean indicating whether to use a GPU for computation.

        Returns:
        - lpips_distance: the LPIPS distance between the two images.
        """
        if self.lpips_fn is None:
            return torch.zeros((img1.shape[0], 1), device=img1.device, dtype=img1.dtype)

        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)

        img1 = img1.float().clamp(0, 1) * 2 - 1
        img2 = img2.float().clamp(0, 1) * 2 - 1

        lpips_fn = self.lpips_fn.to(img1.device)

        # Calculate LPIPS distance
        with torch.no_grad():
            lpips_distance = lpips_fn(img1, img2)

        return lpips_distance
    def forward(self, x, gt_x,tower_idx=0, reuse=False):
        """
        EN: Compute all metrics between prediction x and ground truth gt_x and
        return them as a tuple (mse, ssim, lpips, psnr, pcc, perceptual).

        中文：计算预测 x 与真值 gt_x 之间的全部指标，并以元组
        (mse, ssim, lpips, psnr, pcc, 感知损失) 返回。
        """
        mse = torch.mean(torch.square(gt_x - x)) / 1
        # mse = torch.mean(torch.abs(gt_x - x)) / 20
        ssim = self.SSIM(gt_x, x)
        lpi=self.calculate_lpips(gt_x,x)
        max_pixel=1
        psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        pcc=self.pcc_loss(x,gt_x)
        pc= self.pc(x,gt_x)#perceptual_loss(Hxhat, y)


        return (
            mse,
            ssim,
            lpi,
            psnr_value,
            pcc,
            pc
            # tv_xy,
            # div,
            # pc_loss,
        )

    def __total_variation_2d(self, images):
        pixel_dif2 = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif3 = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        total_var = torch.sum(pixel_dif2) + torch.sum(pixel_dif3)
        return total_var

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

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
    import torch

    def pcc_loss(self,output, target):
        """
        Compute the Pearson Correlation Coefficient (PCC) loss.

        Parameters:
        - output: tensor of predictions from the model.
        - target: tensor of ground truth values.

        Returns:
        - loss: 1 - PCC, where a lower loss indicates a higher correlation between output and target.
        """
        x = output - output.mean()
        y = target - target.mean()
        loss = 1 - (x * y).sum() / (torch.sqrt((x ** 2).sum()) * torch.sqrt((y ** 2).sum()))
        return loss

def render(args):
    """
    渲染函数：基于真实的RI_gt和locations生成相应的图片
    """
    print("Starting render function...")
    
    # EN: Prepare the render output directories and dump the arguments for reproducibility.
    # 中文：准备渲染输出目录，并保存参数文件以便复现。
    # Create output directory
    basedir = args.basedir
    expname = args.object_category_ori+'_render'
    exp_dir = get_experiment_dir(args, expname)
    preview_dir = get_preview_dir(exp_dir)
    ri_dir = get_ri_dir(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(ri_dir, exist_ok=True)
    
    # Save args
    f = os.path.join(exp_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    # Load ground truth data
    print("Loading ground truth data...")
    data_path = args.dataset_path
    
    # EN: Load the ground-truth RI volume and rescale it into the physical RI range [n_b, n_b + max_ri].
    # 中文：加载真值 RI 体数据，并将其重缩放到物理折射率范围 [n_b, n_b + max_ri]。
    # Load RI_gt
    RI_gt_path = Path(data_path + args.data_name + f'/RI_gt.npy')
    if RI_gt_path.exists():
        RI_gt = np.load(RI_gt_path)
        print(np.max(RI_gt), np.min(RI_gt))
        print(f"Loaded RI_gt from {RI_gt_path}, shape: {RI_gt.shape}")
        # Normalize RI_gt similar to training code
        print('min', np.min(RI_gt), 'max', np.max(RI_gt))
        RI_gt = (RI_gt- np.min(RI_gt)) / (np.max(RI_gt) - np.min(RI_gt)) * args.max_ri +args.n_b
        # RI_gt = RI_gt +args.n_b
        print('min', np.min(RI_gt), 'max', np.max(RI_gt))

        # for i in range(RI_gt.shape[2]):
        #     ri_tmp=RI_gt[:,:,i]
        #     ri_tmp=(ri_tmp-np.min(ri_tmp)) / (np.max(ri_tmp) - np.min(ri_tmp)) * 255
        #     cv2.imshow('ri_tmp',ri_tmp.astype('uint8'))
        #     cv2.waitKey(1)
    else:
        print(f"RI_gt file not found at {RI_gt_path}")
        return
    
    # EN: Load the ground-truth illumination source locations that define each view.
    # 中文：加载定义每个视角的照明光源真值位置。
    # Load locations
    locations_path = Path(data_path + args.data_name + '/new_location1024org.npy')
    if locations_path.exists():
        light_loc_gt = np.load(locations_path)
        print(f"Loaded locations from {locations_path}, shape: {light_loc_gt.shape}")
    else:
        print(f"Locations file not found at {locations_path}")
        return
    
    ids = light_loc_gt.shape[0]
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # EN: Build the NPRF forward model (multi-slice beam propagation) with the GT locations, in eval mode.
    # 中文：使用真值光源位置构建 NPRF 正向模型（多层切片光束传播），并置于评估模式。
    # Create NPRF model with ground truth data
    nerf = NPRF(args, locations=light_loc_gt)
    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))
    nerf = nerf.to(device)
    nerf.eval()  # Set to evaluation mode
    
    # Convert RI_gt to tensor
    RI_gt_tensor = torch.tensor(RI_gt).to(device)
    print('min', torch.min(RI_gt_tensor), 'max', torch.max(RI_gt_tensor))
    # Set batch size
    batch = min(args.batch if hasattr(args, 'batch') else 20, 60)
    print(f"Using batch size: {batch}")
    
    # Initialize output array
    imgs = np.zeros((ids, args.grid_x, args.grid_y), dtype=np.float32)
    
    print("Starting rendering process...")
    
    # EN: Batched forward simulation: propagate light through the GT RI volume to synthesize intensity images.
    # 中文：分批正向仿真：让光穿过真值 RI 体数据传播，合成强度图像。
    # Render images in batches
    with torch.no_grad():  # No gradient computation needed for rendering
        for iter in range(0, (ids + batch - 1) // batch):  # Ceiling division
            start_idx = iter * batch
            end_idx = min(start_idx + batch, ids)
            light_loc_ids = np.array(range(start_idx, end_idx))
            
            print(f"Rendering batch {iter + 1}/{(ids + batch - 1) // batch}, "
                  f"indices {start_idx} to {end_idx - 1}")
            
            # Call NPRF forward with ground truth RI
            with torch.autograd.set_detect_anomaly(True):
                # Use the RI_gt directly for rendering
                RI, intensity_pred, index_pred, locations_calibration = nerf(
                    light_loc_ids, 
                    training=False, 
                    steps=0,
                    mask=None,
                    ri_path=RI_gt_tensor  # Pass ground truth RI
                )
            
            # Convert to numpy
            intensity_pred = intensity_pred.cpu().detach().numpy().astype(np.float32)
            # print(intensity_pred.shape)
            # print(imgs.shape)
            # Store rendered images
            imgs[light_loc_ids] = intensity_pred
            
            # Optional display during rendering
            if args.show_img or 1:
                for i, global_idx in enumerate(light_loc_ids):
                    img_pred = intensity_pred[i]
                    img_min = np.min(img_pred)
                    img_max = np.max(img_pred)
                    img_mean = np.mean(img_pred)
                    
                    # Normalize for display
                    if img_max > img_min:
                        img_display = ((img_pred - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img_display = np.zeros_like(img_pred, dtype=np.uint8)
                    
                    # Convert to color for text overlay
                    img_color = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
                    
                    # Add statistics text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img_color, f'Frame: {global_idx:04d}', (10, 30), 
                                font, 0.6, (0, 255, 0), 2)
                    cv2.putText(img_color, f'Min: {img_min:.4f}', (10, 60), 
                                font, 0.6, (0, 255, 0), 2)
                    cv2.putText(img_color, f'Max: {img_max:.4f}', (10, 90), 
                                font, 0.6, (0, 255, 0), 2)
                    cv2.putText(img_color, f'Mean: {img_mean:.4f}', (10, 120), 
                                font, 0.6, (0, 255, 0), 2)
                    
                    # cv2.imshow('rendering_progress', cv2.resize(img_color, None, fx=0.5, fy=0.5))
                    cv2.imwrite(os.path.join(preview_dir, f'img_render{global_idx}.png'), img_color)
                    # cv2.waitKey(1)
                
            
    
    # EN: Save the full synthesized image dataset back next to the source data, then render a preview video.
    # 中文：将完整的合成图像数据集保存回源数据目录，随后生成预览视频。
    # Save complete image dataset
    output_path = Path(data_path + args.data_name + '/new_img1024org.npy')
    np.save(output_path, imgs)
    print(f"Saved rendered images to {output_path}")
    
    # Create video from rendered images
    print("Creating video from rendered images...")
    video_path = Path(data_path + args.data_name + '/new_img1024org.mp4')
    create_video_with_stats(imgs, str(video_path), fps=10)
    
    # Close any open windows
    if args.show_img:
        cv2.destroyAllWindows()
    
    print(f"Rendering complete! Generated {ids} images")
    print(f"Images saved to: {output_path}")
    print(f"Video saved to: {video_path}")

def train(args):
    """
    EN: Main training routine. Loads intensity images and illumination source
    locations, builds the NPRF model (trainable RI volume + multi-slice
    beam-propagation forward model) and a two-group Adam optimizer, then
    iterates: render predicted intensities, normalize and mask them, compute
    losses (data term + TV/L1 regularizers), back-
    propagate, and periodically save checkpoints, RI snapshots, previews,
    videos, and TensorBoard logs. Supports resuming from checkpoints,
    coarse-to-fine stage switching, and self-calibration of source locations
    and voxel pitch.

    中文：主训练流程。加载强度图像与照明光源位置，构建 NPRF 模型
    （可训练 RI 体 + 多层切片光束传播正向模型）以及双参数组 Adam 优化器，
    然后迭代执行：渲染预测强度图、归一化并施加掩膜、计算损失
    （数据项 + TV/L1 正则项）、反向传播，并周期性保存
    断点、RI 快照、预览图、视频与 TensorBoard 日志。支持断点续训、
    由粗到细阶段切换，以及光源位置与体素间距的自校准。
    """

    # Create log dir and copy the config file
    expname = args.object_category_ori
    exp_dir = get_experiment_dir(args, expname)
    preview_dir = get_preview_dir(exp_dir)
    ri_dir = get_ri_dir(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(ri_dir, exist_ok=True)

    # EN: Set up the TensorBoard writer and dump all arguments to args.txt for reproducibility.
    # 中文：初始化 TensorBoard 记录器，并将全部参数写入 args.txt 以便复现。
    #tensorboard
    tensorboard_dir = get_tensorboard_dir(exp_dir)
    args.tbdir = tensorboard_dir
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(Path(tensorboard_dir))
    #writer.add_scalar('Loss/train', 11, 1)
    #writer.close() log_dir
    #metrics_np=np.zeros((8,args.N_iters))

    # save args
    f = os.path.join(exp_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    test_metric_file = os.path.join(exp_dir, 'test_metrics.txt')

    # EN: Data loading: read intensity images and source locations, randomly subsample sub_num views,
    #     and (for simulation datasets) load and rescale the ground-truth RI volume.
    # 中文：数据加载：读取强度图像和光源位置，随机抽取 sub_num 个视角；
    #      若为仿真数据集，还会加载并重缩放真值 RI 体数据。
    # Load data
    print("Start to load data.", end='  ')
    if 1:
        # load data including images and illuminate source location
        args.data_path=args.dataset_path+args.data_name
        images, light_loc_gt = load_phase_data(args.data_path,calib=0)
        #if we use subimage number, we need to select the subimage
        if args.sub_num>images.shape[0]:
            args.sub_num=images.shape[0]
        args.batch = args.sub_num if args.sub_num<20 else 20
        selected_item=np.random.choice(images.shape[0],args.sub_num,replace=False)
        images=images[selected_item]
        light_loc_gt=light_loc_gt[selected_item]
        ids = images.shape[0]
        #if we use the silumation data, we need to load the RI_gt
        if args.simulation:
            RI_gt=np.load(Path(args.data_path+f'/RI_gt.npy'))
            RI_gt =RI_gt-1.33
            print(np.max(RI_gt), np.min(RI_gt))
            RI_gt=RI_gt/np.max(RI_gt)*args.max_ri
        else:
            RI_gt=0
        # RI_gt=0
        if args.simulation:
            print('Loaded phase_data', images.shape, light_loc_gt.shape, RI_gt.shape)
        else:
            print('Loaded phase_data', images.shape, light_loc_gt.shape, 'RI_gt=0 (real data)')

        # if args.show_img:
        #     for i in range(images.shape[0]):
        #         img=images[i]
        #         # print(sampled_items[i], np.mean(img))
        #         #norm
        #         img=img/np.max(img)*255
        #         img=cv2.resize(img,(512,512))
        #         img=img.astype('uint8')
        #         cv2.imshow('img',img)
        #         cv2.waitKey(10)
    else:
        print('Unknown camera dataset type', args.camera_dataset_type, 'exiting')
        return
    shuffle_idx = np.random.permutation(images.shape[0])

    # EN: Optionally corrupt the measurements with Poisson (shot) noise to test robustness.
    # 中文：可选地对测量图像加入泊松（散粒）噪声，以测试鲁棒性。
    if args.add_noise:
        print('Adding img noise')
        for i in range(images.shape[0]):
            img = images[i, :, :]
            img = np.random.poisson(img.astype(np.float32) * args.add_noise) / args.add_noise
            images[i, :, :] = img
            if args.show_img:   
                img=img/np.max(img)*255
                img=cv2.resize(img,(512,512))
                img=img.astype('uint8')
                cv2.imshow('img',img)
                cv2.waitKey(10)
    else:
        print('No adding img noise')
            

    # EN: Initialize (optionally noise-perturbed) source locations and voxel pitch;
    #     these become trainable when self-calibration is enabled.
    # 中文：初始化（可选加噪扰动的）光源位置与体素间距；
    #      启用自校准时这些量将作为可训练参数被优化。
    light_loc = build_location_initializer(light_loc_gt, args)
    dxyz_init = build_dxyz_initializer(args)
    args.dx_init = float(dxyz_init[0])
    args.dy_init = float(dxyz_init[1])
    args.dz_init = float(dxyz_init[2])
    print("Done")
    #add noise to the light location
    

    # EN: Resolve the coarse-to-fine schedule (stage start steps and grid resolutions).
    # 中文：解析由粗到细的训练计划（各阶段起始步数与网格分辨率）。
    global_step = 0
    stage_steps, stage_resolutions = get_stage_schedule(args)
    args.init_block = stage_resolutions[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # EN: Discover existing checkpoints (.tar) in the experiment directory for possible resuming.
    # 中文：在实验目录中查找已有的断点文件（.tar），用于可能的续训。
    # Load Checkpoints
    ckpts = [os.path.join(exp_dir, f) for f in sorted(os.listdir(exp_dir)) if '.tar' in f]
    print('Found ckpts', ckpts)

    # Make the optimizer start with the dataset-specific base LR from step 0.
    get_lr_schedule(args)
    args.block_size = get_stage_resolution(global_step, stage_steps, stage_resolutions)
    # EN: Build the NPRF model at the first-stage resolution, wrap it in DataParallel,
    #     set calibration trainability, and create the two-group optimizer.
    # 中文：以第一阶段分辨率构建 NPRF 模型，使用 DataParallel 封装，
    #      设置自校准参数的可训练状态，并创建双参数组优化器。
    nerf = NPRF(args,locations=light_loc)
    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))
    calibration_active = set_calibration_trainability(nerf, args, global_step)

    optimizer = build_optimizer(args, nerf)

    # EN: Resume from the latest checkpoint (model and optimizer state) unless disabled.
    # 中文：除非禁用重载，否则从最新断点恢复（模型与优化器状态）。
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        global_step = ckpt['global_step']
        netwrok_dict=ckpt['network_state_dict']
        netwrok_dict['module.dxyz']
        # ckpt['network_state_dict']['module.dxyz'][0]=0.25
        # ckpt['network_state_dict']['module.zz'][0]=73

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        smart_load_state_dict(nerf, ckpt)
        ri_pred_path = os.path.join(exp_dir, 'RI_pred.npy')
        if not os.path.exists(ri_pred_path):
            ri_pred_path = os.path.join(args.data_path, 'RI_pred.npy')
        RI_pre = np.load(ri_pred_path)
        light_loc = light_loc_gt


    # EN: Instantiate the composite training loss and move model, loss, and GT volume to the GPU.
    # 中文：实例化复合训练损失，并将模型、损失和真值体数据移动到 GPU。
    loss=Loss()

    # Move data to GPU
    nerf = nerf.to(device)
    loss = loss.to(device)
    RI_gt=torch.tensor(RI_gt).to(device)


    # EN: Main training loop: one mini-batch of views per iteration for N_iters iterations.
    # 中文：主训练循环：共 N_iters 次迭代，每次处理一个小批量视角。
    #begin training
    print('Begin')  
    batch = args.batch
    nerf.train()
    i_batch = 0
    for iter in range(0, args.N_iters):

        print(global_step, ':', end=' ')
        # EN: Self-calibration is activated lazily once global_step reaches self_calibration_step.
        # 中文：当 global_step 达到 self_calibration_step 后才延迟启用自校准。
        wants_calibration = _bool_flag(getattr(args, "position_calibration_enable", 0)) or _bool_flag(getattr(args, "dxyz_calibration_enable", 0))
        if wants_calibration and (not calibration_active) and global_step >= int(getattr(args, "self_calibration_step", 0)):
            calibration_active = set_calibration_trainability(nerf, args, global_step)
            print(f"Enable self calibration at step {global_step}", end=' ')
        # EN: Reshuffle the view order after each epoch, then fetch the next mini-batch
        #     of (source locations, measured intensities, view indices).
        # 中文：每个 epoch 结束后重新打乱视角顺序，然后取出下一个小批量
        #      （光源位置、测量强度图、视角索引）。
        if (i_batch+1) * batch >= ids:
            #print("Shuffle data after an epoch!")
            shuffle_idx = np.random.permutation(ids)
            i_batch = 0
        #shuffle_idx = np.array(range(ids))
        light_loc_training, intensity, light_loc_ids = process_traning_data_simu(images, light_loc_gt,shuffle_idx, i_batch, batch)
        temp_a=intensity.mean()
        intensity = torch.tensor(intensity).cuda().float()


        # EN: Core optimization step: forward render through the trainable volume, build losses, and update.
        # 中文：核心优化步骤：通过可训练体做正向渲染、构建损失并更新参数。
        #####  Core optimization loop  #####
        with (torch.autograd.set_detect_anomaly(True)):


            light_loc_ids_t = torch.tensor(light_loc_ids).long().cuda()
            # EN: Forward pass: NPRF returns the current RI volume, predicted intensity images,
            #     predicted indices, and the (possibly calibrated) source locations.
            # 中文：前向传播：NPRF 返回当前 RI 体、预测强度图、预测索引，
            #      以及（可能经过校准的）光源位置。
            RI, intensity_pred, index_pred, locations_calibration = nerf(
                light_loc_ids_t,
                steps=global_step,
                steps_c2f=stage_steps,
                block_sizes=stage_resolutions,
            )
            # DataParallel gather concatenates non-batch outputs along dim 0;
            # take only the first replica for RI and locations.
            # 中文：DataParallel 的 gather 会将非批次维的输出沿第 0 维拼接；
            #      这里只保留第一个副本的 RI 与光源位置。
            if args.num_gpu > 1:
                ri_size = int(RI.shape[0] // args.num_gpu)
                RI = RI[:ri_size]
                loc_size = int(locations_calibration.shape[0] // args.num_gpu)
                locations_calibration = locations_calibration[:loc_size]


            # intensity_pred = 1 * intensity_pred.permute(1, 2, 0) / (
            #     torch.max(intensity.view(intensity.size(0), -1), dim=1).values)
            # intensity_pred = intensity_pred.permute(2, 0, 1)
            
            # intensity = 1 * intensity.permute(1, 2, 0) / (
            #     torch.max(intensity.view(intensity.size(0), -1), dim=1).values)
            # intensity = intensity.permute(2, 0, 1)

            # EN: Build per-view circular masks centered on each source's image-plane projection,
            #     and collect per-image min/max and mean/std statistics for normalization.
            # 中文：为每个视角构建以光源在像面投影为圆心的圆形掩膜，
            #      并统计每张图的最小/最大值与均值/标准差，用于后续归一化。
            radius = args.radius
            mask_batch = torch.zeros(images[light_loc_ids].shape).float()
            max_min_intensity_pred = torch.zeros(intensity_pred.shape[0],2)
            max_min_intensity = torch.zeros(intensity.shape[0],2)
            mean_std_intensity_pred = torch.zeros(intensity_pred.shape[0],2)
            mean_std_intensity = torch.zeros(intensity.shape[0],2)
            for i in range(intensity_pred.shape[0]):


                loc_x = int((light_loc_training[i][0] + 0.5) * mask_batch.shape[1])
                loc_y = int((light_loc_training[i][1] + 0.5) * mask_batch.shape[2])
                mask_cpu = (255 * mask_batch[i]/torch.max(mask_batch[i])).cpu().detach().numpy().astype('uint8')
                mask_cpu = cv2.circle(mask_cpu, (int(loc_x), int(loc_y)), radius, (255, 255, 155), -1)
                mask_batch[i] = torch.tensor(mask_cpu / 255).float().cuda()

                min_value=torch.min(intensity_pred[i])
                max_value=torch.max(intensity_pred[i])
                max_min_intensity_pred[i] = torch.tensor([min_value,max_value])
                mean_std_intensity_pred[i] = torch.tensor([intensity_pred[i].mean(),intensity_pred[i].std()])

                min_value=torch.min(intensity[i])
                max_value=torch.max(intensity[i])
                max_min_intensity[i] = torch.tensor([min_value,max_value])
                mean_std_intensity[i] = torch.tensor([intensity[i].mean(),intensity[i].std()])

            #normalize by std and mean
            # intensity_pred = ((intensity_pred.permute(1, 2, 0) - mean_std_intensity_pred[:,0]) / ( mean_std_intensity_pred[:,0])).permute(2, 0, 1)
            # intensity = ((intensity.permute(1, 2, 0) - mean_std_intensity[:,0]) / ( mean_std_intensity[:,0])).permute(2, 0, 1)



            



            # mean = intensity_pred[mask_batch == 1].mean()
            # std = intensity_pred[mask_batch == 1].std()
            # intensity_pred[mask_batch == 1] = (intensity_pred[mask_batch == 1] - mean) / (std + 1e-5)
            # intensity_pred = intensity_pred * mask_batch

            #normalize the intensity_pred
            # min_value=torch.min(intensity_pred.view(intensity_pred.size(0), -1), dim=1)
            # max_value=torch.max(intensity_pred.view(intensity_pred.size(0), -1), dim=1)
            # intensity_pred = (intensity_pred.permute(1, 2, 0) - min_value.values) / (max_value.values - min_value.values)
            # intensity_pred = intensity_pred.permute(2, 0, 1)
            #calculate the mean of intensity_pred for each channel
            # mean = intensity_pred.mean(dim=(1,2))
            # print(mean)
            # print(intensity_pred.shape,mask_batch.shape)
            # intensity_pred = intensity_pred * mask_batch
            #set the first pixel to 1
            # intensity_pred[:,0,0]=1e-6


            # mean = intensity[mask_batch == 1].mean()
            # std = intensity[mask_batch == 1].std()
            # intensity[mask_batch == 1] = (intensity[mask_batch == 1] - mean) / (std + 1e-5)
            # intensity = intensity * mask_batch
            
            # min_value=torch.min(intensity.view(intensity.size(0), -1), dim=1)
            # max_value=torch.max(intensity.view(intensity.size(0), -1), dim=1)
            # intensity = (intensity.permute(1, 2, 0) - min_value.values) / (max_value.values - min_value.values)
            # intensity = intensity.permute(2, 0, 1)
            # intensity = intensity * mask_batch

            # EN: Normalize predicted and measured intensities with the selected mode
            #     ('minmax', 'mean_std', or 'std_minmax', all using GT statistics), then apply the masks.
            # 中文：按所选模式（'minmax'、'mean_std' 或 'std_minmax'，均使用真值统计量）
            #      对预测与测量强度图做归一化，然后乘以掩膜。
            # normalization
            norm_mode = getattr(args, 'norm_mode', 'minmax')
            if norm_mode == 'mean_std':
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                intensity_pred = intensity_pred * mask_batch
                intensity = ((intensity.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                intensity = intensity * mask_batch
            elif norm_mode == 'std_minmax':
                # step 1: std normalization (using gt stats)
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                intensity = ((intensity.permute(1, 2, 0) - mean_std_intensity[:,0]) / (mean_std_intensity[:,1] + 1e-5)).permute(2, 0, 1)
                # step 2: minmax normalization (per-image, using gt min/max after std)
                gt_min = intensity.view(intensity.size(0), -1).min(dim=1).values
                gt_max = intensity.view(intensity.size(0), -1).max(dim=1).values
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - gt_min) / (gt_max - gt_min + 1e-5)).permute(2, 0, 1)
                intensity_pred = intensity_pred * mask_batch
                intensity = ((intensity.permute(1, 2, 0) - gt_min) / (gt_max - gt_min + 1e-5)).permute(2, 0, 1)
                intensity = intensity * mask_batch
            else:  # minmax
                intensity_pred = ((intensity_pred.permute(1, 2, 0) - max_min_intensity[:,0]) / (max_min_intensity[:,1] - max_min_intensity[:,0])).permute(2, 0, 1)
                intensity_pred = intensity_pred * mask_batch
                intensity = ((intensity.permute(1, 2, 0) - max_min_intensity[:,0]) / (max_min_intensity[:,1] - max_min_intensity[:,0])).permute(2, 0, 1)
                intensity = intensity * mask_batch

            # intensity_pred = intensity_pred / torch.sum(intensity_pred, (1,2)).unsqueeze(1).unsqueeze(1)*torch.sum(intensity, (1,2)).unsqueeze(1).unsqueeze(1)
            # intensity = intensity
            # print('total energy pred', torch.sum(intensity_pred, (1,2)))
            # print('total energy gt', torch.sum(intensity, (1,2)))

            # EN: Write per-view preview PNGs of the prediction and optionally display GT vs. prediction
            #     side by side with the source location marked.
            # 中文：为每个视角保存预测预览 PNG；可选地并排显示真值与预测图，并标出光源位置。
            for i in range(intensity_pred.shape[0]):
                # print(light_loc_ids[i])
                # tifffile.imwrite('./RI_pred/RI_3D.tif', RI.detach().cpu().numpy().transpose(2,1,0), photometric='minisblack')

                mask=mask_batch[i].cpu().detach().numpy()
                img_pred = (intensity_pred[i]).cpu().detach().numpy()
                img_pred=(255*(img_pred - np.min(img_pred))/(np.max(img_pred)-np.min(img_pred)))
                img_pred=(img_pred*1).astype('uint8')
                img_gt = (intensity[i]).cpu().detach().numpy()
                img_gt = (255*(img_gt - np.min(img_gt))/(np.max(img_gt)-np.min(img_gt)))
                img_gt = (img_gt * 1).astype('uint8')
                img_pred=cv2.resize(img_pred,img_gt.shape)
                cv2.imwrite(os.path.join(preview_dir, f'img_pred{shuffle_idx[i+batch*i_batch]}.png'), img_pred)
                loc_x = int((light_loc_training[i][0] + 0.5) * img_gt.shape[0])
                loc_y = int((light_loc_training[i][1] + 0.5) * img_gt.shape[1])
                img_gt_temp = cv2.circle(img_gt,(int(loc_x),int(loc_y)), radius, (255, 255, 155), 7)
                img_pred_temp = cv2.circle(img_pred,(int(loc_x),int(loc_y)), radius, (255, 255, 155), 7)
                if args.show_img:
                    #cv2.imshow('mask', cv2.resize(mask_cpu, None, fx=1, fy=1).astype('uint8'))
                    img3 = np.hstack((img_gt_temp, img_pred_temp))
                    # img3=255*(img3-np.min(img3))/(np.max(img3)-np.min(img3)).astype('uint8')
                    cv2.imshow('img_gt&pred', cv2.resize(img3, None, fx=0.5, fy=0.5))
                    cv2.waitKey(10)


            # EN: Coarse-to-fine stage switch: rebuild the optimizer when a new stage starts.
            # 中文：由粗到细阶段切换：进入新阶段时重建优化器。
            if _bool_flag(args.c2f_enable) and global_step in stage_steps[1:]:
                optimizer = build_optimizer(args, nerf)


            # EN: Downsample intensities, masks, and predictions to the current stage resolution
            #     so the loss is computed at the coarse-to-fine scale.
            # 中文：将强度图、掩膜与预测图下采样到当前阶段分辨率，
            #      使损失在由粗到细的对应尺度上计算。
            block_size = get_stage_resolution(global_step, stage_steps, stage_resolutions)
            intensity = intensity.unsqueeze(0)
            downsampled = F.interpolate(intensity, size=(block_size*1,block_size*1),mode='bilinear', align_corners=False)
            intensity = downsampled.squeeze(0)

            mask_batch = mask_batch.unsqueeze(0)
            downsampled = F.interpolate(mask_batch, size=(block_size*1,block_size*1),mode='bilinear', align_corners=False)
            mask_batch = downsampled.squeeze(0)

            intensity_pred = intensity_pred.unsqueeze(0)
            downsampled = F.interpolate(intensity_pred, size=(block_size*1,block_size*1),mode='bilinear', align_corners=False)
            intensity_pred = downsampled.squeeze(0)




            #mse = img2mse(intensity_pred, intensity) * 3
            #ssim_loss = 1 - compute_img_metric(intensity_pred * 100, intensity * 100, 'ssim')
            # EN: Compute the composite training loss plus diagnostic quantities
            #     (L1 regularizer, location error vs. GT, RI error vs. GT).
            # 中文：计算复合训练损失及诊断量（L1 正则、相对真值的光源位置误差、
            #      相对真值的 RI 误差）。
            l1_reg=torch.mean(torch.norm(RI-1.33,p=1,dim=2))/50
            locations_mse=torch.mean(abs(locations_calibration-torch.tensor(light_loc_gt)))
            ri_mse=torch.mean(abs(RI-RI_gt)**2)
            #print(locations_mse)
            losses, mse, tv_z, ssim , tv_xy,div,pc,MSE_ri,mse1,mse2= loss(args, intensity_pred, RI, intensity, global_step,xhat_gt=RI_gt)
            #losses+=l1_reg
            #ri_loss=0#img2mse(RI[:,:,1:],RI_gt)*10
            # print(intensity_pred.dtype, intensity.dtype, torch.max(intensity_pred[0]), torch.max(intensity[0]))
            #loss = mse + ssim_loss+ri_loss

            print(f'loss mse,ssim,MSE_ri,mse_location,mse_ri, {losses.item():05f},{mse.item():05f}, {ssim.item():05f}, {locations_mse:05f},{ri_mse:05f}',end=' ')
            print(f'tv_xy,tv_z: {tv_xy.item():05f},{tv_z.item():05f}')
            #re print these values

            with open(test_metric_file, 'a') as file:
                 file.write(f'{global_step:04d}:loss mse,ssim,MSE_ri {losses.item():5f}, {mse.item():05f}, {ssim.item():05f},;'
                            f'tv_xy,tv_z: {tv_xy.item():05f},{tv_z.item():05f}\n')

            # EN: Backpropagate, apply the staged learning-rate schedule to both parameter groups
            #     (locations scaled proportionally), and take the optimizer step.
            # 中文：反向传播，按分段学习率计划更新两个参数组的学习率
            #      （光源位置组按比例缩放），然后执行优化器步进。
            optimizer.zero_grad()
            losses.backward()
            weight_loc = 1 if _bool_flag(args.self_calibration_enable) else 0

            # NOTE: IMPORTANT!
            ##   update learning rate   ###
            # decay_rate = 0.1
            # decay_steps = args.lrate_decay * 1000
            # new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            new_lrate = get_main_learning_rate(args, global_step)
            lr_scale = new_lrate / args.lrate if args.lrate > 0 else 1.0
            for param_group in optimizer.param_groups:
                if param_group.get('group_name') == 'locations':
                    param_group['lr'] = args.position_lrate * lr_scale
                else:
                    param_group['lr'] = new_lrate
            # param=optimizer.param_groups[0]['params'][0]
            # param.grad=param.grad*100
            # grad=param.grad/1000
            # grad.sum()
            optimizer.param_groups[0]['capturable'] = True
            optimizer.step()
            global_step += 1


        # EN: Adaptive save/checkpoint intervals: save more frequently early in training,
        #     unless explicit overrides are provided.
        # 中文：自适应保存/断点间隔：训练早期保存更频繁，除非显式指定覆盖值。
        ########################################################
        if getattr(args, "i_save_override", 0) > 0:
            args.i_save = args.i_save_override
        elif global_step<100:
            args.i_save=10
        elif global_step<1000:
            args.i_save=100
        else:
            args.i_save=200

        if getattr(args, "i_weights_override", 0) > 0:
            args.i_weights = args.i_weights_override
        elif global_step<100:
            args.i_weights=20
        elif global_step<1000:
            args.i_weights=100
        else:
            args.i_weights = 200
        # EN: Periodically save a training checkpoint (model and optimizer state).
        # 中文：周期性保存训练断点（模型与优化器状态）。
        if global_step % args.i_weights == 0:
            path = os.path.join(exp_dir, '{:06d}.tar'.format(global_step))
            torch.save({
                'global_step': global_step,
                'network_state_dict': nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # EN: Snapshot dump: save the current RI volume (npy/tif), calibrated locations and dxyz,
        #     per-slice RI images, predicted/GT image pairs, and comparison videos.
        # 中文：快照保存：写出当前 RI 体（npy/tif）、校准后的光源位置与 dxyz、
        #      逐切片 RI 图像、预测/真值图像对以及对比视频。
        should_save_snapshot = (global_step % args.i_save == 0 and global_step > 0) or (iter == args.N_iters - 1)
        if should_save_snapshot:
            print('Start save', end=' ')
            # print('dxyz=', optimizer.param_groups[0]['params'][1].data)
            path = os.path.join(exp_dir, str(global_step))
            if not os.path.exists(path):
                os.makedirs(path)
            RI_cpu = RI.detach().cpu().numpy()
            locations=locations_calibration.detach().cpu().numpy()
            dxyz = nerf.module.dxyz.detach().cpu().numpy()
            if tifffile is not None:
                ri_tif = np.transpose(RI_cpu.astype(np.float32), (2, 0, 1))
                tifffile.imwrite(os.path.join(path, 'ri.tif'), ri_tif)
                tifffile.imwrite(os.path.join(ri_dir, 'ri.tif'), ri_tif)
            np.save(os.path.join(exp_dir, 'locations_calib.npy'), locations)
            np.save(os.path.join(exp_dir, 'dxyz_calib.npy'), dxyz)
            np.save(os.path.join(exp_dir, 'RI_pred.npy'), RI_cpu)
            np.save(path + '/locations.npy', locations)
            np.save(path + '/dxyz.npy', dxyz)
            np.save(path + '/RI.npy', RI_cpu)
            np.save(os.path.join(ri_dir, 'RI.npy'), RI_cpu)

            # EN: If a ground-truth RI volume exists, evaluate the reconstruction
            #     (MSE/SSIM/LPIPS/PSNR/PCC) and append the results to the metric logs.
            # 中文：若存在真值 RI 体数据，则评估重建质量
            #      （MSE/SSIM/LPIPS/PSNR/PCC），并将结果追加到指标日志中。
            #########evaluate the RI#########
            if os.path.exists(args.data_path+'/RI_gt.npy'):
                # from run_nerf_evaluate import metrics
                print('Evaluate the RI')
                ri_gt=np.load(args.data_path+'/RI_gt.npy')
                ri_gt=(ri_gt-np.min(ri_gt))/(np.max(ri_gt)-np.min(ri_gt))
                # ri_gt=ri_gt.transpose(2,0,1)
                # ri_gt=ri_gt/np.max(ri_gt)
                ri_pred=RI_cpu
                ri_pred[ri_pred<0]=0
                ri_gt[ri_gt<0]=0
                ri_pred=ri_pred/np.max(ri_pred)

                print(ri_pred.shape, np.max(ri_pred), np.min(ri_pred))
                print(ri_gt.shape, np.max(ri_gt), np.min(ri_gt))

                ri_pred = torch.tensor(ri_pred).permute(2, 0, 1).unsqueeze(1).float()
                ri_gt = torch.tensor(ri_gt).permute(2, 0, 1).unsqueeze(1).float()

                try:
                    mse,ssim,lpi,psnr,pcc,pc=metrics()(ri_pred,ri_gt)
                    mse=mse
                    print(f'mse:{mse:.4e},ssim:{ssim:.4f},lpips:{lpi.mean().item():.4f},psnr:{psnr.item():.4f},pcc:{pcc.item():.4f},pc:{pc.item():.4f}')
                    with open(path+'/metrics.txt','a') as f:
                        f.write(f'RI: mse:{mse:.4e},ssim:{ssim:.4f},lpips:{lpi.mean().item():.4f},psnr:{psnr.item():.4f},pcc:{pcc.item():.4f},pc:{pc.item():.4f}\n')
                    with open(get_summary_metrics_path(exp_dir), 'a') as f:
                        f.write(f'{args.object_category_ori}:')
                        f.write(f'RI: mse:{mse:.4e},ssim:{ssim:.4f},lpips:{lpi.mean().item():.4f},psnr:{psnr.item():.4f},pcc:{pcc.item():.4f},pc:{pc.item():.4f}\n')
                except Exception as exc:
                    print(f'Skip RI metric evaluation: {exc}')
                # ri_gt=ri_gt.permute(1,2,3,0).cpu().numpy().squeeze()-1.33
                # ri_pred=ri_pred.permute(1,2,3,0).cpu().numpy().squeeze()-1.33
                # generate_video_ri(ri_gt,ri_pred,path)


            for i in range(RI_cpu.shape[2]):
                img = RI_cpu[:, :, i]
                img_norm = (img-np.min(img)) / (np.max(img)-np.min(img)) * 255
                cv2.imwrite(os.path.join(path, 'RI_pred_' + str(i)) + '.png', img_norm.astype('uint8'))
                if args.show_img:
                    cv2.imshow('ri',img_norm.astype('uint8'))
                    cv2.waitKey(10)
            for i in range(intensity_pred.shape[0]):
                img_pred = (255 * intensity_pred[i]/torch.max(intensity_pred[i])).cpu().detach().numpy().astype('uint8')
                img_gt = (255 * intensity[i]/torch.max(intensity[i])).cpu().detach().numpy().astype('uint8')
                img3 = np.hstack((img_gt, img_pred))
                cv2.imwrite(os.path.join(path, f'img_{str(shuffle_idx[i+batch*i_batch])}_pred.png'), img_pred)
                cv2.imwrite(os.path.join(path, f'img_{str(shuffle_idx[i+batch*i_batch])}_gt.png'), img_gt)
                # img_pred = (255 * intensity_pred[i]/torch.max(intensity_pred[i])).cpu().detach().numpy().astype('uint8')
                # img_gt = (255 * intensity[i]/torch.max(intensity[i])).cpu().detach().numpy().astype('uint8')
                # img3 = np.hstack((img_gt, img_pred))
                # cv2.imwrite(os.path.join(path, str(shuffle_idx[i+batch*i_batch])) + '.png', img_pred)
                if args.show_img:
                    cv2.imshow('img_gt&pred', cv2.resize(img3, None, fx=0.3, fy=0.3))
                    cv2.waitKey(1)
            print('start save video')
            img_cpu=intensity.detach().cpu().numpy()
            img_pred_cpu=intensity_pred.detach().cpu().numpy()
            #stack in the third dimension
            img_comp=np.concatenate((img_cpu,img_pred_cpu),axis=2)
            video_generate(img_comp,path,data_type='img')
            #for ri
            RI_cpu=RI_cpu.transpose(2,0,1)
            RI_cpu=RI_cpu/np.max(RI_cpu)
            if args.simulation:
                RI_GT=RI_gt.detach().cpu().numpy().transpose(2,0,1)
                RI_GT=RI_GT/np.max(RI_GT)
                ri_comp=np.concatenate((RI_GT,RI_cpu),axis=2)
            else:
                ri_comp=RI_cpu
            video_generate(ri_comp,path,data_type='ri')   
        if global_step % args.i_testset == 0 and i > 0:
            print('Start test')
            ########## to be completed ##########
            

        # EN: Log scalar losses, regularizers, calibration errors, and current dx/dz to TensorBoard.
        # 中文：将各项损失、正则项、校准误差以及当前 dx/dz 记录到 TensorBoard。
        if global_step % args.i_tensorboard == 0:
            writer.add_scalar("all/Loss", losses.item(), global_step)
            writer.add_scalar("all/ssim", ssim.item(), global_step)
            #writer.add_scalar("all/MSE_ri*1e6", (MSE_ri * 1e6).item(), global_step)
            writer.add_scalar("MSE/mse", mse.item(), global_step)
            writer.add_scalar("MSE/mse1", mse1.item(), global_step)
            writer.add_scalar("MSE/mse2", mse2.item(), global_step)
            writer.add_scalar("MSE/locations_mse", locations_mse.item(), global_step)
            writer.add_scalar("MSE/ri_mse", ri_mse.item(), global_step)
            #writer.add_scalar("other/pc", pc.item(), global_step)
            writer.add_scalar("rothereg/tv_z", tv_z.item(), global_step)
            writer.add_scalar("rothereg/tv_xy", tv_xy.item(), global_step)
            writer.add_scalar("other/l1_reg", l1_reg.item(), global_step)
            writer.add_scalar("other/div", div.item(), global_step)
            # writer.add_scalar("other/zz", optimizer.param_groups[0]['params'][2].data.item(), global_step)
            writer.add_scalar("other/dx", nerf.module.dxyz.data[0].item(), global_step)
            writer.add_scalar("other/dz", nerf.module.dxyz.data[2].item(), global_step)
        i_batch += 1

    # EN: Optional final evaluation on the latest snapshot after all iterations finish.
    # 中文：全部迭代结束后，可选地对最新快照执行最终评估。
    if _bool_flag(getattr(args, "final_eval_enable", 0)):
        run_final_evaluation(exp_dir, args.data_path)


# EN: Script entry point: parse CLI arguments, fix random seeds, resolve the training policy
#     and experiment name, optionally run the render stage, then launch training.
# 中文：脚本入口：解析命令行参数、固定随机种子、确定训练策略与实验名称，
#      可选地先执行渲染阶段，然后启动训练。
if __name__ == '__main__':
    from datetime import datetime
    from args import config_parser

    if (torch.cuda.is_available()):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()

    # EN: Fix all random seeds (NumPy / PyTorch / CUDA) for reproducibility.
    # 中文：固定所有随机种子（NumPy / PyTorch / CUDA）以保证可复现性。
    seed = 1121
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # EN: Normalize the self-calibration flags and auto-generate the experiment name if requested.
    # 中文：统一整理自校准开关；若实验名设为 "auto"，则自动生成实验名称。
    apply_training_policy(args)

    if args.object_category_ori == "auto":
        args.object_category_ori = build_experiment_name(args)

    # EN: Optionally synthesize images from the GT volume (render stage) before training.
    # 中文：训练前可选地先从真值体数据合成图像（渲染阶段）。
    if args.render:
        print("Running render stage before training...")
        render(args)

    # EN: Run training and report the total wall-clock time.
    # 中文：执行训练并输出总耗时。
    time1 = datetime.now()
    train(args)
    time2 = datetime.now()
    time_diff = time2 - time1
    print(f"time_total(s): {time_diff.total_seconds()}")#verify

    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
