"""Closed loop with the classical solver: inverse CNO -> velocity field -> FDTD.

For each of the 120 test trajectories, the inverse CNO estimates c from
(u(t=0), u(t=dt)); the FDTD solver then resimulates the trajectory from u(t=0)
and the cumulative relative L2 error against the ground truth is reported.
Run from the repository root:  python -m cno_closed_loop.closed_loop.resimulate_fdtd
"""

import os
import torch
import time
import numpy as np
import json
import matplotlib.pyplot as plt

from cno_closed_loop import paths
from cno_closed_loop.inverse.evaluate import make_test_dataset_from_ckpt
from seismic_CPML_2D_pressure_second_order import seismicCPML2D  # vendored FDTD solver (external/seismic_cpml)


def get_cno_prediction(model, dataset, traj_id, dt_gap):
    """Inverse-CNO estimate of c from (u(t=0), u(t=dt_gap)) of a test trajectory, plus its ground truth.

    Uses the module-level `device` set in __main__.
    """
    sample_idx = traj_id * 120 + dt_gap - 1 
    time_val, input_tensor, target_c_norm = dataset[sample_idx]
    
    time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_norm = model(input_batch, time_tensor)
    
    c_pred_norm = prediction_norm[0, 0].cpu().numpy()
    c_gt_norm = target_c_norm[0].cpu().numpy()
    
    # Ground-truth trajectory, all 16 snapshots, in physical units
    i = sample_idx // dataset.multiplier
    h5_idx = i + dataset.start
    u_gt_traj_phys = dataset.reader['u'][h5_idx]

    return c_pred_norm, c_gt_norm, u_gt_traj_phys, dataset.data_path


def denormalize_velocity(c_norm, data_path):
    """Normalized velocity field -> m/s, clipped to the range of the dataset."""
    json_path = os.path.join(data_path, "new_normalization_stats.json")
    with open(json_path, 'r') as f:
        stats = json.load(f)

    c_phys = (c_norm * stats['std_c']) + stats['mean_c']
    c_phys = np.clip(c_phys, stats['min_c'], stats['max_c'])

    return c_phys


def run_physics_validation(c_phys, u0_phys):
    """FDTD resimulation on the 128x128 grid (dx = 40 m) from u0 with velocity c; returns the 16 stored snapshots."""
    NX, NY = 128, 128
    DELTAX, DELTAY = 40, 40 # 20 m at 256x256 -> 40 m at 128x128
    DELTAT = 0.002 # Same dt as generation
    d_steps = 40
    n_steps = 15
    NSTEPS = n_steps * d_steps + 1 # 601 steps total
    NPOINTS_PML = 10
    f0 = 0.5
    dtype = np.float32

    density = np.ones((NX, NY), dtype=dtype)
    # The solver expects (past_pressure, present_pressure).
    # As in the data generation, u0 is passed for both time levels (the wavefield starts at rest).
    initial_pressures = (u0_phys.astype(dtype), u0_phys.astype(dtype))

    wavefields, _ = seismicCPML2D(
        NX=NX, NY=NY, NSTEPS=NSTEPS,
        DELTAX=DELTAX, DELTAY=DELTAY, DELTAT=DELTAT,
        NPOINTS_PML=NPOINTS_PML,
        velocity=c_phys.astype(dtype),
        density=density,
        initial_pressures=initial_pressures,
        f0=f0, dtype=dtype,
        output_wavefields=True, gather_is=None
    )

    # Keep every 40th step, as in the data generation (16 snapshots)
    u_resimulated_trajectory = wavefields[::d_steps]

    return u_resimulated_trajectory


def plot_trajectories_side_by_side(u_gt_traj, u_resim_traj, save_path):
    """Ground-truth (rows 0-3) and resimulated (rows 4-7) snapshots on a shared colour scale."""
    # Shared colour scale for both trajectories
    u_min = min(np.min(u_gt_traj), np.min(u_resim_traj))
    u_max = max(np.max(u_gt_traj), np.max(u_resim_traj))
    
    fig, axes = plt.subplots(8, 4, figsize=(20, 32))
    
    # Ground truth (rows 0-3)
    for i in range(16):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        im = ax.imshow(u_gt_traj[i], vmin=u_min, vmax=u_max, cmap='jet')
        ax.set_title(f"GT u(t={i})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Resimulation (rows 4-7)
    for i in range(16):
        row, col = (i // 4) + 4, i % 4
        ax = axes[row, col]
        im = ax.imshow(u_resim_traj[i], vmin=u_min, vmax=u_max, cmap='jet')
        ax.set_title(f"Resim u(t={i})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def calculate_and_plot_errors(u_gt_traj, u_resim_traj, save_path):
    """Per-step and cumulative relative L2 error of one resimulated trajectory (printed and plotted)."""
    eps = 1e-8
    step_errors = []
    cum_num, cum_den = 0.0, 0.0
    cum_errors = []

    print("\nRelative L2 error per step:")
    print(f"{'Step':<5} | {'Step Error':<12} | {'Cum. Error':<12}")
    print("-" * 35)

    for t in range(16):
        gt_t = u_gt_traj[t]
        resim_t = u_resim_traj[t]
        
        diff_sq = np.sum((resim_t - gt_t)**2)
        gt_sq = np.sum(gt_t**2)
        
        # Step-by-step Rel L2
        step_err = np.sqrt(diff_sq) / (np.sqrt(gt_sq) + eps)
        step_errors.append(step_err)
        
        # Cumulative Rel L2
        cum_num += diff_sq
        cum_den += gt_sq
        cum_err = np.sqrt(cum_num) / (np.sqrt(cum_den) + eps)
        cum_errors.append(cum_err)
        
        print(f"{t:<5} | {step_err:<12.4f} | {cum_err:<12.4f}")
        
    total_error = cum_errors[-1]
    print("-" * 35)
    print(f"Cumulative relative L2 error at the last step: {total_error:.4f}\n")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(16), step_errors, label='Step-by-Step Error', marker='o', color='#1f77b4')
    plt.plot(range(16), cum_errors, label='Cumulative Error', marker='s', color='#ff7f0e')
    plt.xlabel("Time Step")
    plt.ylabel("Relative L2 Error")
    plt.title("Numerical Resimulation Error over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Error plot saved to: {save_path}")
    return total_error


if __name__ == "__main__":
    dt = 9                         # the inverse CNO sees the snapshot pair (t = 0, t = dt)
    num_test_trajectories = 120
    SAVE_TRAJECTORY_PLOTS = False  # per-trajectory figures as in the report appendix (2 PDFs each)
    all_cum_errors = []

    print(f"Closed loop with FDTD: {num_test_trajectories} test trajectories, dt = {dt}")
    start_time = time.time()

    model, dataset = make_test_dataset_from_ckpt(model_name=paths.INVERSE_RUN, checkpoint_name=paths.INVERSE_CKPT, num_traj=num_test_trajectories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    out_dir = str(paths.RESULTS_DIR / "closed_loop")
    os.makedirs(out_dir, exist_ok=True)

    for traj in range(num_test_trajectories):
        # Inverse-CNO estimate of c, in m/s for the solver
        c_pred_norm, c_gt_norm, u_gt_traj_phys, dat_path = get_cno_prediction(
            model=model,
            dataset=dataset,
            traj_id=traj,
            dt_gap=dt
        )
        c_pred_phys = denormalize_velocity(c_norm=c_pred_norm, data_path=dat_path)
        
        # FDTD resimulation from the true initial wavefield
        u0_phys = u_gt_traj_phys[0]
        u_resim_trajectory = run_physics_validation(c_phys=c_pred_phys, u0_phys=u0_phys)
        
        # Cumulative relative L2 error over time
        eps = 1e-8
        diff_sq_per_step = np.sum((u_resim_trajectory - u_gt_traj_phys)**2, axis=(1, 2))  # [16]
        gt_sq_per_step = np.sum(u_gt_traj_phys**2, axis=(1, 2))
        cum_diff_sq = np.cumsum(diff_sq_per_step)
        cum_gt_sq = np.cumsum(gt_sq_per_step)
        traj_cum_errors = np.sqrt(cum_diff_sq) / (np.sqrt(cum_gt_sq) + eps)
        all_cum_errors.append(traj_cum_errors)

        print(f"Trajectory {traj+1:03d}/{num_test_trajectories}: final cumulative L2 error {traj_cum_errors[-1]:.4f}")

        if SAVE_TRAJECTORY_PLOTS:
            plot_trajectories_side_by_side(
                u_gt_traj=u_gt_traj_phys,
                u_resim_traj=u_resim_trajectory, 
                save_path=f"{out_dir}/resimulation_vs_gt_traj{traj}_dt{dt}.pdf"
            )
            calculate_and_plot_errors(
                u_gt_traj=u_gt_traj_phys, 
                u_resim_traj=u_resim_trajectory, 
                save_path=f"{out_dir}/resimulation_errors_traj{traj}_dt{dt}.pdf"
            )
    
    all_cum_errors = np.array(all_cum_errors) # [120, 16]
    mean_cum_errors = np.mean(all_cum_errors, axis=0)
    std_cum_errors = np.std(all_cum_errors, axis=0)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nDone in {elapsed:.1f} min. Final cumulative relative L2 error: {mean_cum_errors[-1]:.4f} ± {std_cum_errors[-1]:.4f}")

    # Mean cumulative error over time with a 1-std band across the trajectories
    plt.figure(figsize=(8, 5))
    steps = np.arange(16)
    plt.plot(steps, mean_cum_errors, label='Mean Cumulative Error', color='#ff7f0e', linewidth=2, marker='o')
    plt.fill_between(steps, 
                     mean_cum_errors - std_cum_errors, 
                     mean_cum_errors + std_cum_errors, 
                     color='#ff7f0e', alpha=0.2, label='± 1 Std Dev')
    
    plt.xlabel("Time Step")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Average Cumulative Error over Time (dt={dt}, N={num_test_trajectories})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = f"{out_dir}/average_cum_error_dt{dt}.pdf"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_path}")
