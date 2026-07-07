"""
EN: Command-line argument parser for FDT training, preconfigured for the UC Davis
dataset (ucdavis_dx0.33). Defaults here encode the acquisition geometry (grid size,
voxel pitch, slice count), the physical imaging model (wavelength, NA, background
refractive index), the training schedule, and optional features such as
self-calibration and coarse-to-fine training.
run_nerf.py consumes the resulting namespace directly.

中文：FDT 训练的命令行参数解析器，默认针对 UC Davis 数据集（ucdavis_dx0.33）预先
配置。这里的默认值编码了采集几何（网格尺寸、体素间距、层数）、物理成像模型
（波长、数值孔径、背景折射率）、训练日程，以及自校准、由粗到细训练等可选功能。
run_nerf.py 直接使用解析得到的参数命名空间。
"""

import argparse


# EN: Build and return the argparse parser holding all FDT training options; call
# .parse_args() on the result to obtain the configuration namespace.
# 中文：构建并返回包含全部 FDT 训练选项的 argparse 解析器；对返回值调用
# .parse_args() 即可得到配置命名空间。
def config_parser():
    parser = argparse.ArgumentParser(description="FDT training config for UCDavis dataset.")

    # EN: Dataset paths and identity — where the data lives, which sub-dataset to use,
    # whether it is a simulation, and how many measurements to subsample.
    # 中文：数据集路径与标识——数据存放位置、使用哪个子数据集、是否为仿真数据，
    # 以及子采样多少张测量图像。
    # Dataset
    parser.add_argument("--dataset_path", default="./dataset/")
    parser.add_argument("--data_name", default="ucdavis_dx0.33")
    parser.add_argument("--object_category_ori", default="auto")
    parser.add_argument("--simulation", action="store_true", default=True)
    parser.add_argument("--sub_num", type=int, default=1500)
    parser.add_argument("--camera_dataset_type", default="ucdavis")

    # EN: Logging and output locations — checkpoint/log directories, TensorBoard
    # directory, optional run tag, and checkpoint-reload behavior.
    # 中文：日志与输出位置——检查点/日志目录、TensorBoard 目录、可选的运行标签，
    # 以及是否重新加载已有检查点。
    # Logging / outputs
    parser.add_argument("--basedir", default="./log")
    parser.add_argument("--tbdir", default="./log/tensorboard")
    parser.add_argument("--txt", default="")
    parser.add_argument("--no_reload", action="store_true", default=False)

    # EN: Training schedule — iteration count, learning rates and their staged/decayed
    # schedules, batch size, GPU count, and the periodic save/eval/log intervals.
    # 中文：训练日程——迭代次数、学习率及其分阶段/衰减策略、批大小、GPU 数量，
    # 以及周期性保存、评估和日志记录的间隔。
    # Training schedule
    parser.add_argument("--N_iters", type=int, default=1000)
    parser.add_argument("--lrate", type=float, default=5e-3)
    parser.add_argument("--position_lrate", type=float, default=1e-3)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--lr_stage_steps", nargs="+", type=int, default=[150, 500, 750])
    parser.add_argument("--lr_stage_values", nargs="+", type=float, default=[5e-3, 5e-3, 5e-3, 5e-3])
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--i_save", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=100)
    parser.add_argument("--i_save_override", type=int, default=0)
    parser.add_argument("--i_weights_override", type=int, default=0)
    parser.add_argument("--i_testset", type=int, default=1000)
    parser.add_argument("--i_tensorboard", type=int, default=10)

    # EN: Runtime toggles, synthetic noise, and self-calibration — inject noise into
    # light-source positions or voxel spacing (dxyz) to test robustness, and enable
    # learned correction (self-calibration) of those quantities during training;
    # also includes coarse-to-fine (c2f) resolution staging and final evaluation.
    # 中文：运行时开关、合成噪声与自校准——向光源位置或体素间距（dxyz）注入噪声以
    # 测试鲁棒性，并可在训练中对这些量进行可学习的校正（自校准）；此组还包含
    # 由粗到细（c2f）的分辨率分阶段策略与最终评估开关。
    # Runtime toggles
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--show_img", action="store_true", default=False)
    parser.add_argument("--add_noise", type=float, default=0.0)
    parser.add_argument("--location_noise_enable", type=int, default=0)
    parser.add_argument("--location_noise_std", type=float, default=0.1)
    parser.add_argument("--location_noise_scale", type=float, default=0.1)
    parser.add_argument("--location_noise_seed", type=int, default=1121)
    parser.add_argument("--dxyz_noise_enable", type=int, default=0)
    parser.add_argument("--dxyz_noise_std", type=float, default=0.1)
    parser.add_argument("--dxyz_noise_scale", type=float, default=0.1)
    parser.add_argument("--dxyz_noise_seed", type=int, default=2203)
    parser.add_argument("--disable_train_location_noise", type=int, default=1)
    parser.add_argument("--self_calibration_enable", type=int, default=0)
    parser.add_argument("--position_calibration_enable", type=int, default=0)
    parser.add_argument("--dxyz_calibration_enable", type=int, default=0)
    parser.add_argument("--self_calibration_step", type=int, default=100)
    parser.add_argument("--c2f_enable", type=int, default=0)
    parser.add_argument("--c2f_stage_steps", nargs="+", type=int, default=[0, 200, 400, 600])
    parser.add_argument("--c2f_stage_resolutions", nargs="+", type=int, default=[128, 256, 512, 512])
    parser.add_argument("--final_eval_enable", type=int, default=1)

    # EN: Reconstruction geometry matched to the ucdavis_dx0.33 dataset — lateral grid
    # size, number of axial slices, voxel pitches (dx/dy/dz, in microns), free-space
    # propagation distance (fs), measurement count, and illumination ring radius.
    # 中文：与 ucdavis_dx0.33 数据集匹配的重建几何——横向网格尺寸、轴向层数、
    # 体素间距（dx/dy/dz，单位微米）、自由空间传播距离（fs）、测量数量以及
    # 照明环半径。
    # Dataset / geometry matched to ucdavis_dx0.33
    parser.add_argument("--grid_x", type=int, default=512)
    parser.add_argument("--grid_y", type=int, default=512)
    parser.add_argument("--layers", type=int, default=14)
    parser.add_argument("--dx", type=float, default=0.33)
    parser.add_argument("--dy", type=float, default=0.33)
    parser.add_argument("--dz", type=float, default=1.5)
    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--n_measure", type=int, default=1500)
    parser.add_argument("--radius", type=int, default=2400)

    # EN: Physical model of the optical system — illumination wavelength, numerical
    # aperture, background (medium) refractive index, maximum RI contrast, physical
    # field-of-view size, and back-to-back / back-to-center geometry offsets.
    # 中文：光学系统的物理模型——照明波长、数值孔径、背景（介质）折射率、最大折射率
    # 对比度、物理视场尺寸，以及 back-to-back / back-to-center 几何偏移量。
    # Physical model
    parser.add_argument("--wavelength", type=float, default=0.6)
    parser.add_argument("--NA", type=float, default=0.65)
    parser.add_argument("--n_b", type=float, default=1.33)
    parser.add_argument("--max_ri", type=float, default=0.03)
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=168.96)
    parser.add_argument("--W", type=float, default=168.96)
    parser.add_argument("--b2b", type=float, default=0.0)
    parser.add_argument("--b2c", type=float, default=0.0)

    # EN: Volume representation — the explicit trainable voxel grid ('exp'), its initial
    # coarse resolution, and the LeakyReLU slope used when mapping raw values to RI.
    # 中文：体表示——显式可训练体素网格（'exp'）、其初始粗分辨率，以及将原始数值
    # 映射为 RI 时使用的 LeakyReLU 斜率。
    # Representation
    parser.add_argument("--model", default="exp")
    parser.add_argument("--init_block", type=int, default=128)
    parser.add_argument("--patch_ratio", type=float, default=1.0)
    parser.add_argument("--relu_slope", type=float, default=0.2)

    # EN: Loss and regularization — data-fidelity loss type, intensity normalization
    # mode, total-variation weights in the lateral (xy) and axial (z) directions, and
    # DnCNN denoiser normalization range.
    # 中文：损失与正则化——数据保真损失类型、强度归一化方式、横向（xy）与轴向（z）
    # 的全变分权重，以及 DnCNN 去噪器的归一化范围。
    # Loss / regularization
    parser.add_argument("--loss", default="l12")
    parser.add_argument("--norm_mode", default="std_minmax", help="normalization: mean_std, minmax, std_minmax")
    parser.add_argument("--regularize_type", default="")
    parser.add_argument("--tv_xy", type=float, default=0.0)
    parser.add_argument("--tv_z", type=float, default=0.0)
    parser.add_argument("--DnCNN_normalization_min", type=float, default=0.0)
    parser.add_argument("--DnCNN_normalization_max", type=float, default=1.0)

    return parser
