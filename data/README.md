# Data

The dataset is not included in the repository (about 5.8 GB). The code expects
two files in `data/`, or in the folder given by `CNO_CL_DATA_DIR`:

| File | Content |
|---|---|
| `new_wave_gauss_data.h5` | `u`: float32 `[5256, 16, 128, 128]`, wavefield snapshots at t = 0, …, 15 (physical units)<br>`c`: float32 `[5256, 128, 128]`, velocity field in m/s |
| `new_normalization_stats.json` | `mean_u`, `std_u`, `min_u`, `max_u`, `mean_c`, `std_c`, `min_c`, `max_c`, `means_per_step`, `stds_per_step`, `time_steps` (15), `n_steps` (15) |

**Splits** (as in the CNO dataset classes): the last 120 trajectories are the
test set, the 30 before them the validation set, and the rest are used for training.

## How the data were generated

Wave-Gauss problem (report, Section 3): the 2D acoustic wave equation
∂²u/∂t² = c(x, y)² ∇²u in a stationary, heterogeneous medium.

- **Medium:** a base velocity plus four randomly placed Gaussian inclusions.
- **Initial condition:** two to six Gaussian pulses, starting at rest (∂u/∂t = 0).
- **Solver:** second-order FDTD with C-PML absorbing boundaries (the solver in
  [`external/seismic_cpml`](../external/seismic_cpml)) on a 256 × 256 grid plus
  3 cells of padding per side, with dx = 20 m, dt = 2 ms and 601 time steps.
- **Output:** every 40th step is stored (16 snapshots). The fields are then
  cropped and spectrally downsampled to 128 × 128.

The generator script (by Bogdan Raonić, parallelized for this project) is not
included in this repository.
