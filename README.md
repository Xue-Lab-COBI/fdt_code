# FDT — Fluorescence Diffraction Tomography using Explicit Neural Fields

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

## System requirements

### Software

The code is written in Python and depends on the packages listed in
`requirements.txt` (`torch>=2.0`, `torchvision`, `numpy`, `opencv-python`,
`matplotlib`, `tifffile`, `tensorboard`, `absl-py`). Any recent Linux
distribution with Python ≥ 3.10 and a CUDA-enabled PyTorch build should work.

The code has been tested with the following environment:

| Component | Tested version |
| --- | --- |
| OS | Rocky Linux 8.10 (Linux kernel 4.18) |
| Python | 3.12.12 |
| CUDA driver / toolkit | 580.126.20 / CUDA 13.0 |
| torch | 2.13.0+cu130 |
| torchvision | 0.28.0 |
| numpy | 2.3.5 |
| opencv-python | 5.0.0.93 |
| matplotlib | 3.10.8 |
| tifffile | 2026.3.3 |
| tensorboard | 2.21.0 |
| absl-py | 2.5.0 |

### Hardware

No non-standard hardware is required, but a CUDA-capable NVIDIA GPU is
required for training. The results in the paper were produced on an NVIDIA
RTX A6000 (48 GB). The demo below has also been verified on an RTX 2080 Ti
(11 GB); peak GPU memory usage during training is about 9 GB.

## Installation

```bash
pip install -r requirements.txt
```

Typical install time: about 2 minutes on a normal desktop computer with a
broadband connection (measured: 131 s in a clean Python 3.12 environment,
dominated by the PyTorch/CUDA wheel downloads).

## Demo (quick start)

1. **Generate the synthetic dataset** (written under `./dataset/ucdavis_dx0.33/`):

   ```bash
   python test_ri_ucdavis_gen.py --save-root dataset
   ```

   Expected output: `RI_gt.npy` (ground-truth RI volume, 512×512×14),
   `new_location1024org.npy` (1500 illumination positions), a `.tif` copy of
   the volume, a layer-preview `.mp4`, and `parameters.json`.
   Typical run time: under 1 second.

2. **Train** (use `--render 1` on the first run so the measurement stack
   `new_img1024org.npy` is rendered from the ground truth; later runs can
   reuse it with `--render 0`):

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u run_nerf.py \
     --N_iters 500 \
     --render 1 \
     --fs 50 \
     --max_ri 0.03 \
     --location_noise_enable 1 \
     --self_calibration_enable 1 \
     --c2f_enable 1 \
     --c2f_stage_steps 0 10 30 50 \
     --c2f_stage_resolutions 128 256 512 512
   ```

   Expected output: a per-iteration loss log on stdout, and a run directory
   `./log/<exp_name>/` (see [Outputs](#outputs)) containing the reconstructed
   volume `RI_pred.npy` and final metrics. With this configuration the final
   evaluation printed at the end should be approximately
   `mse ≈ 4e-03, ssim ≈ 0.59, psnr ≈ 24` (exact values vary slightly between
   runs because mini-batches are shuffled).
   Typical run time: about 15 minutes for 500 iterations on an RTX 2080 Ti
   (measured: 926 s including one-time rendering of the 1500-image
   measurement stack); an RTX A6000 is comparable or faster.

3. **Evaluate / export colormapped slices**:

   ```bash
   python test_ri_ucdavis_eval.py --ri-path ./log/<exp_name>/RI_pred.npy
   ```

   Expected output: per-slice statistics on stdout and colormapped BMP images
   of each z-slice (using `colormap0627.npy`). Typical run time: a few
   seconds.

## Running on your own data

Training reads two `.npy` files from `<dataset_path>/<data_name>/`
(defaults: `./dataset/ucdavis_dx0.33/`, see `load_data.py`):

- `new_img1024org.npy` — measured intensity stack, float array of shape
  `(N, H, W)` (one image per illumination source).
- `new_location1024org.npy` — illumination source positions, float array of
  shape `(N, 3)`, in the same normalized coordinate convention as
  `generate_light_positions()` in `test_ri_ucdavis_gen.py`.

Steps:

1. Convert your measurements and calibrated source positions to the two
   `.npy` files above and place them in `./dataset/<your_name>/`.
2. Adjust the geometry / physics arguments to match your system (see
   `args.py`): `--grid_x --grid_y --layers` (reconstruction grid),
   `--dx --dy --dz` (voxel pitch, µm), `--wavelength --NA --n_b --max_ri`,
   `--fs` (free-space propagation distance), and `--n_measure --sub_num`
   (number of measurements).
3. Train with `--data_name <your_name> --render 0` (`--render 1` is only for
   simulation, where the stack is rendered from `RI_gt.npy`). For real data,
   enabling `--self_calibration_enable 1` is recommended to refine source
   positions and voxel pitch.
4. If a ground-truth volume `RI_gt.npy` exists in the dataset folder, RI
   metrics are computed automatically at the end of training; otherwise only
   the reconstruction is saved.

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

## License

MIT — see [LICENSE](LICENSE).
