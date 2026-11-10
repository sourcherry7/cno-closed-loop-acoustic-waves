"""FDTD error floor on the 128x128 grid: the FDTD baseline driven by the TRUE velocity.

The closed-loop comparison (compare_cno_vs_fdtd.py) runs the FDTD solver on the
128x128 grid (dx = 40 m) with the velocity field predicted by the inverse CNO.
The ground-truth trajectories, however, were simulated on a finer grid
(256x256 plus padding, dx = 20 m) and then downsampled to 128x128. Running the
same 128x128 FDTD setup with the ground-truth velocity separates the solver's
own discretization/boundary error from the error caused by the predicted
velocity field.

Runs on CPU in about 2 minutes:  python -m cno_closed_loop.closed_loop.fdtd_floor
"""
import os

import h5py
import numpy as np

from cno_closed_loop import paths
from cno_closed_loop.closed_loop.compare_cno_vs_fdtd import run_numerical_resimulation


def cumulative_relative_l2(pred, true, eps=1e-8):
    """Cumulative relative L2 error over time, as in compare_cno_vs_fdtd.py ([N, T, H, W] -> [N, T])."""
    cum_diff = np.cumsum(np.sum((pred - true) ** 2, axis=(2, 3)), axis=1)
    cum_true = np.cumsum(np.sum(true ** 2, axis=(2, 3)), axis=1)
    return np.sqrt(cum_diff) / (np.sqrt(cum_true) + eps)


if __name__ == "__main__":
    with h5py.File(paths.DATA_DIR / "new_wave_gauss_data.h5", "r") as f:
        n_total = f["u"].shape[0]
        n_test = int(240 * n_total / 10512.0)        # same test split as the dataset classes
        u_gt = f["u"][n_total - n_test:]             # [120, 16, 128, 128], physical units
        c_gt = f["c"][n_total - n_test:]             # [120, 128, 128], m/s

    print(f"FDTD (128x128) with the true velocity on {n_test} test trajectories...")
    u_fdtd = np.stack([run_numerical_resimulation(c_gt[i], u_gt[i, 0]) for i in range(n_test)])

    err = cumulative_relative_l2(u_fdtd, u_gt)
    mean, std = err.mean(axis=0), err.std(axis=0)
    print("step : " + " ".join(f"{k:6d}" for k in range(len(mean))))
    print("mean : " + " ".join(f"{m:6.3f}" for m in mean))
    print("std  : " + " ".join(f"{s:6.3f}" for s in std))
    print(f"\nFinal step: {100 * mean[-1]:.1f}% +/- {100 * std[-1]:.1f}% cumulative relative L2")

    out_dir = paths.RESULTS_DIR / "closed_loop"
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(out_dir / "fdtd_true_velocity_floor.csv", np.stack([np.arange(len(mean)), mean, std], axis=1),
               delimiter=",", header="step,mean_cum_rel_l2,std_cum_rel_l2", comments="")
    print(f"Saved {out_dir / 'fdtd_true_velocity_floor.csv'}")
