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

## License

MIT — see [LICENSE](LICENSE).
