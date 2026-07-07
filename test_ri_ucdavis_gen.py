"""
EN: Generator for the UC Davis synthetic refractive-index (RI) tomography dataset used by FDT.
It synthesizes a 3D ground-truth RI volume by stamping the letters "UCDavis" onto separate
z-slices, samples a uniform grid of illumination (light source) locations, renders a
layer-by-layer preview video, and writes everything (RI volume, light positions, TIFF stack,
parameter JSON) into a dataset folder.

中文：FDT 项目所用 UC Davis 合成折射率（RI）层析数据集的生成脚本。
脚本将字母 "UCDavis" 分别印制到不同的 z 层，合成三维真值 RI 体数据；
在归一化坐标上均匀采样照明光源位置网格；渲染逐层预览视频；
并将所有结果（RI 体数据、光源位置、TIFF 堆栈、参数 JSON）保存到数据集目录中。
"""

import argparse
import json
import os

import cv2
import numpy as np
import tifffile as tiff


def normalize_to_uint8(img, vmin, vmax):
    """
    EN: Linearly rescale an image from the [vmin, vmax] range to 8-bit [0, 255].
    Values outside the range are clipped; a degenerate range yields an all-zero image.

    中文：将图像从 [vmin, vmax] 区间线性缩放到 8 位 [0, 255] 范围。
    超出区间的数值会被裁剪；若区间无效（vmax <= vmin）则返回全零图像。
    """
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    img_clip = np.clip(img, vmin, vmax)
    img_norm = (img_clip - vmin) / (vmax - vmin)
    return (img_norm * 255.0).astype(np.uint8)


def generate_light_positions(num_positions=1500, grid_size=40):
    """Generate uniformly distributed light source positions in normalized coordinates."""
    light_loc = np.zeros((num_positions, 3), dtype=np.float32)

    # EN: Build a regular grid_size x grid_size lattice over the central [0.25, 0.75] square.
    # 中文：在中心 [0.25, 0.75] 正方形区域内构建 grid_size x grid_size 的规则网格。
    x_values = np.linspace(0.25, 0.75, grid_size)
    y_values = np.linspace(0.25, 0.75, grid_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # EN: Take the first num_positions grid points and shift them so coordinates are
    # centered at zero (range [-0.25, 0.25]); z stays 0 (sources in one plane).
    # 中文：取前 num_positions 个网格点并平移使坐标以零为中心（范围 [-0.25, 0.25]）；
    # z 坐标保持为 0（光源位于同一平面上）。
    light_loc[:, 0] = x_grid.ravel()[:num_positions] - 0.5
    light_loc[:, 1] = y_grid.ravel()[:num_positions] - 0.5
    light_loc[:, 2] = 0.0
    return light_loc


def generate_ucdavis_volume(
    shape_x=512,
    shape_y=512,
    layer=14,
    letters="UCDavis",
    layer_step=2,
    font_scale=10,
    thickness=10,
    target_width=100,
    ri_base=1.33,
    ri_delta=0.03,
):
    """Generate the same UCDavis letter stack as the original script."""
    # EN: Volume synthesis — start from an empty (shape_x, shape_y, layer) stack and place
    # one letter per z-slice, spaced layer_step slices apart.
    # 中文：体数据合成——从空的 (shape_x, shape_y, layer) 堆栈开始，
    # 每隔 layer_step 层放置一个字母（每个 z 层一个字母）。
    image_stack = np.zeros((shape_x, shape_y, layer), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    saved_letter_slices = []
    for letter_index, letter in enumerate(letters):
        z_idx = layer_step * letter_index
        if z_idx >= layer:
            break

        # EN: Rescale the font so each letter has a consistent target width, then draw it
        # centered on a blank image.
        # 中文：调整字体比例使每个字母具有一致的目标宽度，然后将其居中绘制到空白图像上。
        img = np.zeros((shape_x, shape_y, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(letter, font, font_scale, thickness)[0]
        font_scale_adjusted = font_scale * target_width / text_size[1]
        text_size_adjusted = cv2.getTextSize(letter, font, font_scale_adjusted, thickness)[0]
        text_x = (img.shape[1] - text_size_adjusted[0]) // 2
        text_y = (img.shape[0] + text_size_adjusted[1]) // 2

        cv2.putText(
            img,
            letter,
            (text_x, text_y),
            font,
            font_scale_adjusted,
            (255, 255, 255),
            thickness,
        )

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_stack[:, :, z_idx] = img_gray
        saved_letter_slices.append(z_idx)

    # EN: Convert the binary letter mask into refractive-index values: background = ri_base
    # (e.g. water, 1.33) and letter foreground = ri_base + ri_delta.
    # 中文：将二值字母掩码转换为折射率数值：背景为 ri_base（如水的 1.33），
    # 字母前景为 ri_base + ri_delta。
    image_stack = image_stack.astype(np.float32)
    image_stack = image_stack / np.max(image_stack) * ri_delta + ri_base
    return image_stack, saved_letter_slices


def create_ucdavis_video(volume, video_path, fps=3, resize_factor=0.5, ri_base=1.33, ri_delta=0.03):
    """Create a layer-by-layer preview video for the generated UCDavis volume."""
    # EN: Set up an MP4 writer sized to the (optionally downscaled) slice resolution.
    # 中文：按照（可选缩放后的）切片分辨率创建 MP4 视频写入器。
    height, width, num_layers = volume.shape
    display_height = int(height * resize_factor)
    display_width = int(width * resize_factor)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (display_width, display_height))

    print(f"Creating layer video with {num_layers} slices...")
    # EN: For each z-slice: map RI values to 8-bit grayscale, resize, overlay per-slice
    # statistics (min/max/mean/foreground count) as text, and append the frame to the video.
    # 中文：对每个 z 层切片：将 RI 值映射为 8 位灰度图，缩放尺寸，
    # 叠加该层统计信息（最小/最大/均值/前景像素数）文字，并将帧写入视频。
    for z in range(num_layers):
        img_slice = volume[:, :, z]
        img_u8 = normalize_to_uint8(img_slice, vmin=ri_base, vmax=ri_base + ri_delta)
        img_resized = cv2.resize(img_u8, (display_width, display_height), interpolation=cv2.INTER_NEAREST)
        img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        img_min = float(np.min(img_slice))
        img_max = float(np.max(img_slice))
        img_mean = float(np.mean(img_slice))
        img_nonzero = int(np.count_nonzero(img_slice > ri_base))

        cv2.putText(
            img_color,
            f"Z-Layer: {z:02d}/{num_layers - 1:02d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            f"Min/Max: {img_min:.3f}/{img_max:.3f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            f"Mean: {img_mean:.4f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            f"Foreground: {img_nonzero}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            img_color,
            "UCDAVIS",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        writer.write(img_color)

    writer.release()
    print(f"Layer video saved to: {video_path}")


def save_ucdavis_dataset(
    save_root="dataset",
    shape_x=512,
    shape_y=512,
    layer=14,
    dx=0.33,
    dy=0.33,
    dz=1.5,
    letters="UCDavis",
    num_light_positions=1500,
    light_grid_size=40,
):
    """
    EN: End-to-end dataset export pipeline. Synthesizes the ground-truth RI volume,
    saves it as .npy and .tif, samples the illumination locations, renders the
    layer preview video, and records all generation parameters in parameters.json.
    Returns the volume and the dataset directory path.

    中文：端到端的数据集导出流程。合成真值 RI 体数据，保存为 .npy 和 .tif，
    采样照明光源位置，渲染逐层预览视频，并将全部生成参数记录到 parameters.json。
    返回体数据和数据集目录路径。
    """
    # EN: Step 1 — volume synthesis: build the UCDavis letter RI stack.
    # 中文：步骤 1——体数据合成：构建 UCDavis 字母 RI 堆栈。
    volume, saved_letter_slices = generate_ucdavis_volume(
        shape_x=shape_x,
        shape_y=shape_y,
        layer=layer,
        letters=letters,
    )

    name = f"ucdavis_dx{dx}"
    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)

    # EN: Step 2 — save the ground-truth volume as .npy (two aliases) and as a
    # z-first float32 TIFF stack.
    # 中文：步骤 2——将真值体数据保存为 .npy（两个副本文件名）以及
    # z 轴优先的 float32 TIFF 堆栈。
    np.save(os.path.join(save_dir, "RI_gt.npy"), volume)
    np.save(os.path.join(save_dir, "ucdavis.npy"), volume)
    tiff.imwrite(
        os.path.join(save_dir, f"{name}.tif"),
        volume.transpose(2, 0, 1).astype(np.float32),
    )

    # EN: Step 3 — location sampling: generate the illumination positions and save them.
    # 中文：步骤 3——位置采样：生成照明光源位置并保存。
    light_loc = generate_light_positions(
        num_positions=num_light_positions,
        grid_size=light_grid_size,
    )
    np.save(os.path.join(save_dir, "new_location1024org.npy"), light_loc)

    # EN: Step 4 — rendering: write the layer-by-layer preview video of the volume.
    # 中文：步骤 4——渲染：输出体数据的逐层预览视频。
    layer_video_path = os.path.join(save_dir, f"{name}_layers.mp4")
    create_ucdavis_video(volume, layer_video_path, fps=3, resize_factor=0.5)

    # EN: Step 5 — persist all generation parameters/metadata to parameters.json.
    # 中文：步骤 5——将全部生成参数/元数据持久化到 parameters.json。
    params_info = {
        "shape_xyz": [shape_x, shape_y, layer],
        "dx_um": dx,
        "dy_um": dy,
        "dz_um": dz,
        "letters": list(letters),
        "letter_slices": saved_letter_slices,
        "ri_base": 1.33,
        "ri_delta": 0.03,
        "num_light_positions": int(light_loc.shape[0]),
        "light_grid_size": int(light_grid_size),
    }
    with open(os.path.join(save_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(params_info, f, indent=2)

    print(f"volume shape: {volume.shape}")
    print(f"volume dtype: {volume.dtype}")
    print(f"volume min/max: {float(np.min(volume)):.6f}/{float(np.max(volume)):.6f}")
    print(f"letter slices: {saved_letter_slices}")
    print(f"light positions shape: {light_loc.shape}")
    print(f"layer video: {layer_video_path}")
    print(f"saved to: {save_dir}")
    return volume, save_dir


def main():
    """
    EN: Command-line entry point. Parses dataset shape, voxel pitch, letters, output
    location and illumination sampling options, then runs the export pipeline.

    中文：命令行入口。解析数据集尺寸、体素间距、字母内容、输出路径以及
    照明采样等选项，然后运行数据集导出流程。
    """
    parser = argparse.ArgumentParser(description="Generate the UCDavis RI dataset.")
    parser.add_argument("--shape-x", type=int, default=512)
    parser.add_argument("--shape-y", type=int, default=512)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--dx", type=float, default=0.33)
    parser.add_argument("--dy", type=float, default=0.33)
    parser.add_argument("--dz", type=float, default=1.5)
    parser.add_argument("--letters", default="UCDavis")
    parser.add_argument("--save-root", default="dataset")
    parser.add_argument("--num-light-positions", type=int, default=1500)
    parser.add_argument("--light-grid-size", type=int, default=40)
    args = parser.parse_args()

    save_ucdavis_dataset(
        save_root=args.save_root,
        shape_x=args.shape_x,
        shape_y=args.shape_y,
        layer=args.layer,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        letters=args.letters,
        num_light_positions=args.num_light_positions,
        light_grid_size=args.light_grid_size,
    )


if __name__ == "__main__":
    main()
