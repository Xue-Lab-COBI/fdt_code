# FDT — Fluorescence Diffraction Tomography using Explicit Neural Fields

**English** | [中文](#中文说明)

Code for the paper *"Fluorescence Diffraction Tomography using Explicit
Neural Fields"*.

FDT reconstructs a 3D refractive-index (RI) volume from a stack of intensity
measurements. The forward model is a multi-slice beam-propagation simulator
(`optics.py`); the object is represented by an explicit neural field — a
trainable voxel grid (`NeRF.py`) optimized with gradient descent
(`run_nerf.py`). Optional features include self-calibration of illumination
source locations / voxel pitch and coarse-to-fine (multi-resolution)
training.

## Repository layout

| File | Purpose |
| --- | --- |
| `run_nerf.py` | Main training / rendering entry point |
| `optics.py` | Physics forward model: `PhaseObject3D`, `TomographySolver` (multi-slice propagation) |
| `NeRF.py` | Trainable volume representation (explicit voxel grid, coarse-to-fine) |
| `loss.py` | Training losses (L1/L2, SSIM, perceptual, TV regularization) |
| `ssim.py` | SSIM metric implementation |
| `load_data.py` | Dataset loading utilities |
| `args.py` | Argument parser used by `run_nerf.py` (preconfigured for the UC Davis dataset) |
| `test_ri_ucdavis_gen.py` | Generate the UC Davis synthetic dataset (GT RI volume, source locations, rendered intensity stack) |
| `test_ri_ucdavis_eval.py` | Evaluate a reconstructed RI volume and export colormapped slices |
| `launch_from_args_template.py` | Relaunch training from a saved `args.txt` template |
| `colormap0627.npy` | Custom colormap used for visualization |

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended.

## Quick start

1. **Generate a dataset** (written under `./dataset/`):

   ```bash
   python test_ri_ucdavis_gen.py --save-root dataset
   ```

2. **Train**:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u run_nerf.py \
     --N_iters 500 \
     --render 0 \
     --fs 50 \
     --max_ri 0.03 \
     --location_noise_enable 1 \
     --self_calibration_enable 1 \
     --c2f_enable 1 \
     --c2f_stage_steps 0 10 30 50 \
     --c2f_stage_resolutions 128 256 512 512
   ```

3. **Evaluate**:

   ```bash
   python test_ri_ucdavis_eval.py --ri-path ./log/<exp_name>/RI_pred.npy
   ```

## Key arguments

- `--fs` — free-space propagation distance
- `--max_ri` — RI contrast upper bound used for normalization
- `--location_noise_enable` — perturb initial illumination locations
- `--self_calibration_enable` — make source locations / voxel pitch trainable
- `--c2f_enable`, `--c2f_stage_steps`, `--c2f_stage_resolutions` — coarse-to-fine schedule (`c2f_stage_steps` must start with `0`, both lists have equal length)
- `--render` — `1` regenerates the rendered measurement stack, `0` reuses existing data

## Outputs

Each run writes to `./log/<experiment_name>/`, containing `args.txt`,
checkpoints (`*.tar`), TensorBoard logs (`tensorboard/`), training previews
(`image_pred/`), intermediate RI volumes (`RI_pred/`), the latest
reconstruction (`RI_pred.npy`, `locations_calib.npy`), and metrics
(`test_metrics.txt`, `metricsRI.txt`).

---

# 中文说明

本仓库是论文 *"Fluorescence Diffraction Tomography using Explicit Neural
Fields"* 的代码。

FDT 从一组强度图像重建三维折射率（RI）体数据。前向模型是多层
beam-propagation 物理模拟器（`optics.py`）；被重建对象用显式 neural
field——可训练体素网格表示（`NeRF.py`），通过梯度下降优化
（`run_nerf.py`）。可选功能包括光源位置 / 体素间距自校准
（self-calibration）和由粗到细（coarse-to-fine）多分辨率训练。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `run_nerf.py` | 训练 / 渲染主入口 |
| `optics.py` | 物理前向模型：`PhaseObject3D`、`TomographySolver`（多层传播） |
| `NeRF.py` | 可训练体表示（显式体素网格，支持由粗到细） |
| `loss.py` | 训练损失（L1/L2、SSIM、感知损失、TV 正则） |
| `ssim.py` | SSIM 指标实现 |
| `load_data.py` | 数据集读取工具 |
| `args.py` | `run_nerf.py` 使用的参数解析器（针对 UC Davis 数据集预设） |
| `test_ri_ucdavis_gen.py` | 生成 UC Davis 仿真数据集（GT RI 体、光源位置、渲染强度栈） |
| `test_ri_ucdavis_eval.py` | 评估重建 RI 体并导出伪彩色切片 |
| `launch_from_args_template.py` | 从已保存的 `args.txt` 模板重新启动训练 |
| `colormap0627.npy` | 可视化用的自定义 colormap |

## 安装

```bash
pip install -r requirements.txt
```

强烈建议使用支持 CUDA 的 GPU。

## 快速开始

1. **生成数据集**（写入 `./dataset/`）：

   ```bash
   python test_ri_ucdavis_gen.py --save-root dataset
   ```

2. **训练**：

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u run_nerf.py \
     --N_iters 500 \
     --render 0 \
     --fs 50 \
     --max_ri 0.03 \
     --location_noise_enable 1 \
     --self_calibration_enable 1 \
     --c2f_enable 1 \
     --c2f_stage_steps 0 10 30 50 \
     --c2f_stage_resolutions 128 256 512 512
   ```

3. **评估**：

   ```bash
   python test_ri_ucdavis_eval.py --ri-path ./log/<实验名>/RI_pred.npy
   ```

## 常用参数

- `--fs` — free-space 传播距离
- `--max_ri` — 归一化用的 RI 对比度上限
- `--location_noise_enable` — 是否给初始光源位置加噪声
- `--self_calibration_enable` — 光源位置 / 体素间距是否可训练
- `--c2f_enable`、`--c2f_stage_steps`、`--c2f_stage_resolutions` — 由粗到细训练日程（`c2f_stage_steps` 必须以 `0` 开头，两个列表长度相同）
- `--render` — `1` 表示重新渲染测量数据，`0` 表示复用已有数据

## 输出

每次训练写入 `./log/<实验名>/`，包含 `args.txt`、checkpoint（`*.tar`）、
TensorBoard 日志（`tensorboard/`）、训练过程预览（`image_pred/`）、
RI 中间结果（`RI_pred/`）、最新重建结果（`RI_pred.npy`、
`locations_calib.npy`）以及指标文件（`test_metrics.txt`、`metricsRI.txt`）。

## License

MIT — see [LICENSE](LICENSE).
