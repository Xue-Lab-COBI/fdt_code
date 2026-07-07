"""
EN: Relaunch an FDT training run from a saved args.txt template. Each run writes its
full configuration to args.txt ("key = value" lines); this script parses such a file
back into a namespace, applies a few required overrides (free-space distance, dataset
name, experiment name) plus arbitrary --set key=value overrides, then invokes
run_nerf's training (and optional rendering) exactly as the original entry point would.

中文：从已保存的 args.txt 模板重新启动 FDT 训练。每次训练都会把完整配置写入
args.txt（"key = value" 格式）；本脚本把该文件解析回参数命名空间，应用若干必需的
覆盖项（自由空间距离、数据集名称、实验名称）以及任意的 --set key=value 覆盖，
然后像原始入口一样调用 run_nerf 的训练（以及可选的渲染）流程。
"""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import run_nerf


# EN: Convert one textual value from args.txt into a Python object via
# ast.literal_eval, with fallbacks for booleans and raw strings.
# 中文：用 ast.literal_eval 把 args.txt 中的文本值转换为 Python 对象，
# 无法解析时回退处理布尔值或按原始字符串返回。
def parse_value(text: str):
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except Exception:
        if text == "True":
            return True
        if text == "False":
            return False
        return text


# EN: Parse an args.txt file ("key = value" per line) into a dictionary,
# skipping any lines that do not match the expected format.
# 中文：将 args.txt（每行 "key = value"）解析为字典，跳过不符合格式的行。
def load_args_txt(path: Path):
    args_dict = {}
    with path.open() as f:
        for line in f:
            if " = " not in line:
                continue
            key, value = line.rstrip("\n").split(" = ", 1)
            args_dict[key] = parse_value(value)
    return args_dict


# EN: Entry point: read the CLI options, merge the template with required and optional
# overrides, seed all RNGs for reproducibility, apply the training policy, then run
# rendering (if requested) followed by training.
# 中文：主入口：读取命令行选项，将模板与必需及可选覆盖项合并，为可复现性固定所有
# 随机种子，应用训练策略，然后按需先渲染再启动训练。
def main():
    parser = argparse.ArgumentParser(description="Launch FDT run from an existing args.txt template.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--fs", type=float, required=True)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--expname", required=True)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--render", type=int, default=None)
    parser.add_argument("--num-gpu", type=int, default=1)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override template args with key=value. May be provided multiple times.",
    )
    args_cli = parser.parse_args()

    # EN: Merge the template with required overrides (fs, dataset, experiment name,
    # output dirs), then optional N_iters/render values and free-form --set overrides.
    # 中文：先用必需覆盖项（fs、数据集、实验名称、输出目录）更新模板，再应用可选的
    # N_iters/render 取值以及自由形式的 --set 覆盖项。
    args_dict = load_args_txt(Path(args_cli.template))
    args_dict.update(
        {
            "fs": args_cli.fs,
            "data_name": args_cli.data_name,
            "object_category_ori": args_cli.expname,
            "tbdir": f"./log/{args_cli.expname}/tensorboard",
            "num_gpu": args_cli.num_gpu,
            "basedir": "./log",
            "dataset_path": "./dataset/",
        }
    )
    if args_cli.n_iters is not None:
        args_dict["N_iters"] = args_cli.n_iters
    if args_cli.render is not None:
        args_dict["render"] = args_cli.render
    for override in args_cli.overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}. Expected key=value.")
        key, value = override.split("=", 1)
        args_dict[key] = parse_value(value)

    args = SimpleNamespace(**args_dict)

    # EN: Match the original entry point's runtime setup: default CUDA tensors when a
    # GPU is present, and fixed seeds for NumPy/PyTorch reproducibility.
    # 中文：复现原始入口的运行时设置：有 GPU 时默认使用 CUDA 张量，并固定
    # NumPy/PyTorch 随机种子以保证可复现。
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    seed = 1121
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # EN: Apply run_nerf's training policy adjustments, resolve the "auto" experiment
    # name, then launch rendering (if enabled) and training.
    # 中文：应用 run_nerf 的训练策略调整，解析 "auto" 实验名称，随后（若启用）先
    # 执行渲染，再启动训练。
    run_nerf.apply_training_policy(args)
    if args.object_category_ori == "auto":
        args.object_category_ori = run_nerf.build_experiment_name(args)

    if getattr(args, "render", 0):
        run_nerf.render(args)
    run_nerf.train(args)


if __name__ == "__main__":
    main()
