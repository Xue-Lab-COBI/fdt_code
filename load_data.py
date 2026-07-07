"""
EN: Dataset loading and batching utilities for FDT. Loads the pre-rendered intensity
image stack and the corresponding illumination (light source) positions from .npy
files, slices them into shuffled training mini-batches, and can export a 3D volume
as an .mp4 video for quick visual inspection of reconstructions.

中文：FDT 的数据加载与批处理工具。从 .npy 文件读取预先渲染的强度图像堆栈及对应的
照明（光源）位置，按打乱后的索引切分训练小批量，并支持将三维体数据导出为 .mp4
视频，便于快速查看重建结果。
"""

import os

import cv2
import numpy as np


# EN: Load the measured intensity images and their light-source positions from fixed
# .npy filenames under data_path; the calib argument is accepted for interface
# compatibility but ignored. Returns (images, light positions) as float32 arrays.
# 中文：从 data_path 下固定命名的 .npy 文件加载实测强度图像及其光源位置；
# calib 参数仅为接口兼容而保留，实际未使用。返回 float32 的（图像，光源位置）数组。
def load_phase_data(data_path, calib=0):
    del calib
    image_path = os.path.join(data_path, "new_img1024org.npy")
    light_path = os.path.join(data_path, "new_location1024org.npy")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Rendered image stack not found: {image_path}")
    if not os.path.exists(light_path):
        raise FileNotFoundError(f"Light position file not found: {light_path}")

    images = np.load(image_path).astype(np.float32)
    light_loc = np.load(light_path).astype(np.float32)
    return images, light_loc


# EN: Slice one training mini-batch: take the i_batch-th window of the shuffled index
# list and gather the matching light positions and intensity images (the last batch
# may be smaller). Also returns the selected indices for bookkeeping.
# 中文：切分一个训练小批量：取打乱索引列表中第 i_batch 个窗口，据此提取对应的光源
# 位置与强度图像（最后一个批次可能不足 batch 大小），同时返回所选索引以便记录。
def process_traning_data_simu(images, light_loc_gt, shuffle_idx, i_batch, batch):
    start = i_batch * batch
    end = min(start + batch, len(shuffle_idx))
    light_loc_ids = shuffle_idx[start:end]
    light_loc_training = light_loc_gt[light_loc_ids]
    intensity = images[light_loc_ids]
    return light_loc_training, intensity, light_loc_ids


# EN: Write a 3D volume (slices along axis 0) to an .mp4 video, normalizing each slice
# to [0, 255] independently; returns the output path. Non-3D inputs are skipped and
# only the intended path is returned.
# 中文：将三维体数据（沿第 0 轴逐层切片）写入 .mp4 视频，每一层单独归一化到
# [0, 255]，返回输出文件路径。若输入不是三维数组则跳过写入，仅返回目标路径。
def video_generate(volume, folder_path, data_type="img", fps=8):
    os.makedirs(folder_path, exist_ok=True)
    output_path = os.path.join(folder_path, f"{data_type}.mp4")

    if volume.ndim != 3:
        return output_path

    height, width = volume.shape[1], volume.shape[2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame in volume:
        frame = frame.astype(np.float32)
        frame_min = float(np.min(frame))
        frame_max = float(np.max(frame))
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        else:
            frame = np.zeros_like(frame)
        frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR))

    writer.release()
    return output_path
