"""Closed loop: inverse CNO -> velocity field -> forward resimulation, CNO vs. FDTD.

For each of the 120 test trajectories, the inverse CNO estimates c from the pair
(u(t=0), u(t=dt_gap)). The full trajectory is then resimulated from u(t=0) with
(a) the forward CNO (autoregressive rollout, timed on the GPU) and (b) the FDTD
solver, and both are compared with the ground truth (cumulative relative L2).
Run from the repository root:  python -m cno_closed_loop.closed_loop.compare_cno_vs_fdtd
"""

import os
import torch
import time
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

from cno_closed_loop import paths
from cno_closed_loop.forward.model_loader import WaveLoader
from cno_closed_loop.inverse.evaluate import make_test_dataset_from_ckpt
from seismic_CPML_2D_pressure_second_order import seismicCPML2D  # vendored FDTD solver (external/seismic_cpml)


def load_json_stats(data_path):
    """Normalization statistics of the dataset."""
    json_path = os.path.join(data_path, "new_normalization_stats.json")
    with open(json_path, 'r') as f:
        return json.load(f)

def denormalize_c(c_norm, stats):
    """Normalized velocity field -> m/s (no clipping)."""
    return (c_norm * stats['std_c']) + stats['mean_c']

def get_inverse_and_gt(model, dataset, traj_id, dt_gap, device):
    """Inverse-CNO estimate of c from (u(t=0), u(t=dt_gap)) of a test trajectory, plus its ground truth."""
    sample_idx = traj_id * 120 + dt_gap - 1 
    time_val, input_tensor, target_c_norm = dataset[sample_idx]
    
    time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_norm = model(input_batch, time_tensor)
    
    # Normalized prediction, used directly as forward-CNO input
    c_pred_norm_tensor = prediction_norm[:, 0:1] # [1, 1, H, W]
    c_pred_norm_np = c_pred_norm_tensor[0, 0].cpu().numpy()
    c_gt_norm_np = target_c_norm[0].cpu().numpy()
    
    # Ground-truth trajectory (physical units)
    i = sample_idx // dataset.multiplier
    h5_idx = i + dataset.start
    u_gt_traj = dataset.reader['u'][h5_idx]

    return c_pred_norm_tensor, c_pred_norm_np, c_gt_norm_np, u_gt_traj

class CNORollout(torch.nn.Module):
    """
    Autoregressive rollout of the forward CNO that times only the model calls
    (the GPU is synchronized before and after each call).
    """
    def __init__(self, model, steps=15):
        super().__init__()
        self.model = model
        self.steps = steps

    def forward(self, u_norm, c_tensor, t_channel, t_scalar):
        preds = [u_norm]
        curr_u = u_norm
        total_model_math_time = 0.0 # summed duration of the model calls
        
        for _ in range(1, self.steps + 1):
            x_in = torch.cat([curr_u, c_tensor, t_channel], dim=1)
            
            # Timed: the model call only
            torch.cuda.synchronize() # wait for pending GPU work
            t_start = time.time()
            
            pred_out = self.model(x_in, t_scalar)
            
            torch.cuda.synchronize() # wait for the model call to finish
            t_end = time.time()
            
            total_model_math_time += (t_end - t_start)
            
            curr_u = pred_out[:, 0:1]
            preds.append(curr_u)
            
        return torch.stack(preds), total_model_math_time


def run_numerical_resimulation(c_phys, u0_phys):
    """FDTD resimulation on the 128x128 grid (dx = 40 m); returns the 16 stored snapshots."""
    NX, NY = 128, 128
    DELTAX, DELTAY = 40, 40 
    DELTAT = 0.002 
    d_steps = 40
    NSTEPS = 15 * d_steps + 1 
    NPOINTS_PML = 10
    f0 = 0.5
    dtype = np.float32
    # Non-positive velocities would make the solver unstable
    if np.any(c_phys <= 0):
        print("WARNING: Negative/Zero velocities predicted. FDTD may become unstable.")

    density = np.ones((NX, NY), dtype=dtype)
    initial_pressures = (u0_phys.astype(dtype), u0_phys.astype(dtype))

    wavefields, _ = seismicCPML2D(
        NX=NX, NY=NY, NSTEPS=NSTEPS, DELTAX=DELTAX, DELTAY=DELTAY, DELTAT=DELTAT,
        NPOINTS_PML=NPOINTS_PML, velocity=c_phys.astype(dtype), density=density,
        initial_pressures=initial_pressures, f0=f0, dtype=dtype, output_wavefields=True
    )
    return wavefields[::d_steps]

def evaluate_trajectory_accuracies(u_gt_batch, u_num_batch, u_cno_batch, save_path):
    """Mean +/- 1 std of the cumulative relative L2 errors (inputs: [batch, 16, 128, 128]); prints and plots them."""
    eps = 1e-8
    
    def calc_cum_error_stats(pred, true):
        # Squared error and squared norm per snapshot
        diff_sq = np.sum((pred - true)**2, axis=(2, 3)) # Shape: [Batch, 16]
        true_sq = np.sum(true**2, axis=(2, 3))          # Shape: [Batch, 16]
        
        # Accumulated over time
        cum_diff_sq = np.cumsum(diff_sq, axis=1)        # Shape: [Batch, 16]
        cum_true_sq = np.cumsum(true_sq, axis=1)        # Shape: [Batch, 16]
        
        rel_l2 = np.sqrt(cum_diff_sq) / (np.sqrt(cum_true_sq) + eps)
        
        return np.mean(rel_l2, axis=0), np.std(rel_l2, axis=0)

    mean_cno_gt, std_cno_gt = calc_cum_error_stats(u_cno_batch, u_gt_batch)
    mean_num_gt, std_num_gt = calc_cum_error_stats(u_num_batch, u_gt_batch)
    mean_cno_num, std_cno_num = calc_cum_error_stats(u_cno_batch, u_num_batch)

    print("\nCumulative relative L2 error at the last step (mean ± std):")
    print(f"  FDTD vs ground truth: {mean_num_gt[-1]:.4f} ± {std_num_gt[-1]:.4f}")
    print(f"  CNO  vs ground truth: {mean_cno_gt[-1]:.4f} ± {std_cno_gt[-1]:.4f}")
    print(f"  CNO  vs FDTD:         {mean_cno_num[-1]:.4f} ± {std_cno_num[-1]:.4f}")

    plt.figure(figsize=(10, 6))
    steps = np.arange(16)
    
    # Curve 1: FDTD vs GT
    plt.plot(steps, mean_num_gt, label='FDTD vs GT (Physics Baseline)', marker='o', color='green')
    plt.fill_between(steps, mean_num_gt - std_num_gt, mean_num_gt + std_num_gt, color='green', alpha=0.15)
    
    # Curve 2: CNO vs GT
    plt.plot(steps, mean_cno_gt, label='CNO vs GT (End-to-End Surrogate)', marker='s', color='blue')
    plt.fill_between(steps, mean_cno_gt - std_cno_gt, mean_cno_gt + std_cno_gt, color='blue', alpha=0.15)
    
    # Curve 3: CNO vs FDTD
    plt.plot(steps, mean_cno_num, label='CNO vs FDTD (Surrogate Fidelity)', marker='^', color='red', linestyle='--')
    plt.fill_between(steps, mean_cno_num - std_cno_num, mean_cno_num + std_cno_num, color='red', alpha=0.15)
    
    plt.title("Cumulative Relative L2 Error Over Time (Mean ± 1 Std Dev)")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Relative L2 Error")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved accuracy plot to: {save_path}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high') 
    torch.backends.cudnn.benchmark = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_samples = 120   # test trajectories
    dt_gap = 9            # the inverse CNO sees the snapshot pair (t = 0, t = dt_gap)
    num_iterations = 2    # timed forward rollouts, after one warm-up rollout
    PROFILE = False       # print a torch.profiler table per rollout (adds overhead to the timing)

    # Trained inverse and forward models
    inv_model, inv_dataset = make_test_dataset_from_ckpt(
        model_name=paths.INVERSE_RUN,
        checkpoint_name=paths.INVERSE_CKPT,
        num_traj=total_samples
    )
    inv_model.to(device).eval()

    fwd_base = str(paths.FORWARD_CHECKPOINT_DIR / paths.FORWARD_RUN)
    fwd_loader = WaveLoader(
        train_file=os.path.join(fwd_base, "training_properties.txt"),
        net_file=os.path.join(fwd_base, "net_architecture.txt"),
        ckpt_path=os.path.join(fwd_base, paths.FORWARD_CKPT),
        which_name="wave_gauss"
    )
    fwd_model = fwd_loader.load_model()
    stats = load_json_stats(inv_dataset.data_path)

    print("\nCompiling the forward CNO with torch.compile...")
    fwd_model = torch.compile(fwd_model) 
    rollout_module = CNORollout(fwd_model, steps=15).to(device)

    # Inverse CNO: estimated velocity field for every test trajectory
    print(f"Estimating velocity fields for {total_samples} test trajectories...")
    c_pred_tensors, u0_phys_list, c_pred_phys_list, u_gt_traj_list = [], [], [], []
    for traj_id in range(total_samples):
        c_tensor, c_pred_norm, _, u_gt_traj = get_inverse_and_gt(inv_model, inv_dataset, traj_id, dt_gap, device)
        c_pred_tensors.append(c_tensor.squeeze(0)) 
        u0_phys_list.append(u_gt_traj[0])
        c_pred_phys_list.append(denormalize_c(c_pred_norm, stats))
        u_gt_traj_list.append(u_gt_traj)

    full_c_batch = torch.stack(c_pred_tensors).to(device)
    full_u0_batch = np.stack(u0_phys_list)
    u_gt_batch = np.stack(u_gt_traj_list)

    # Forward CNO: batched autoregressive rollout from u(t = 0), timed on the GPU
    print(f"\nForward CNO rollout: batch of {total_samples}, {num_iterations} timed runs after one warm-up")
    mean_u = torch.tensor(stats['mean_u'], device=device)
    std_u = torch.tensor(stats['std_u'], device=device)
    u0_tensor = torch.from_numpy(full_u0_batch).float().to(device).unsqueeze(1)
    u_norm = (u0_tensor - mean_u) / std_u
    
    dt_val = 1.0 / stats['n_steps']
    t_scalar = torch.ones((total_samples, 1), device=device) * dt_val
    t_channel = torch.ones_like(full_c_batch) * dt_val

    total_pure_math_time = 0.0
    u_fwd_cno_batch = None

    for i in tqdm(range(num_iterations + 1), desc="Timed rollouts"):
        if PROFILE:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("CNO_Full_Rollout"):
                    with torch.no_grad():
                        preds_norm, iteration_math_time = rollout_module(u_norm, full_c_batch, t_channel, t_scalar)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            with torch.no_grad():
                preds_norm, iteration_math_time = rollout_module(u_norm, full_c_batch, t_channel, t_scalar)

        # The first rollout is a warm-up (compilation, cuDNN autotuning)
        if i > 0:
            total_pure_math_time += iteration_math_time
        
        # Predictions of the last run, in physical units, for the accuracy comparison
        if i == num_iterations:
            all_phys = (preds_norm * std_u) + mean_u
            u_fwd_cno_batch = all_phys.squeeze(2).transpose(0, 1).cpu().numpy()

    avg_pure_math_time = total_pure_math_time / num_iterations

    # FDTD baseline: the same estimated velocity fields, solved sequentially on the CPU
    print("\nFDTD baseline")
    start_num = time.time()
    u_num_resim_list = [run_numerical_resimulation(c_pred_phys_list[i], u0_phys_list[i]) for i in tqdm(range(total_samples), desc="FDTD")]
    u_num_batch = np.stack(u_num_resim_list)
    num_total_time = time.time() - start_num

    time_per_sample_ms = (avg_pure_math_time / total_samples) * 1000 
    print(f"\nFDTD (sequential, CPU):     {num_total_time:.2f} s for {total_samples} trajectories")
    print(f"Forward CNO (batched, GPU): {avg_pure_math_time:.2f} s ({time_per_sample_ms:.1f} ms per trajectory)")
    print(f"Speed-up:                   {num_total_time / avg_pure_math_time:.1f}x")

    # Accuracy against the ground truth
    out_dir = paths.RESULTS_DIR / "closed_loop"
    os.makedirs(out_dir, exist_ok=True)
    evaluate_trajectory_accuracies(u_gt_batch, u_num_batch, u_fwd_cno_batch, str(out_dir / "accuracy_comparison.pdf"))
