import os
import sys
import torch
import time
import numpy as np
import json
import matplotlib.pyplot as plt

# --- 1. SETUP PATHS ---
PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
MYMODELS_DIR = f"{PROJECT_ROOT}/src/my_models"
EXTERNAL_CNO_DIR = f"{PROJECT_ROOT}/src/external/CNO/CNO2d_time_dependent_&_foundation_model"
sys.path.append(MYMODELS_DIR)
sys.path.append(EXTERNAL_CNO_DIR)

# Custom Imports
from wave_loaders import WaveLoader
from inverse_problem.inv_testing import make_test_dataset_from_ckpt
sys.path.append(f"{PROJECT_ROOT}/src/external/wave_simulation")
from seismic_CPML_2D_pressure_second_order2 import seismicCPML2D

# --- 2. UTILITY FUNCTIONS ---

def load_json_stats(data_path):
    """Unified normalization loader """
    json_path = os.path.join(data_path, "new_normalization_stats.json")
    with open(json_path, 'r') as f:
        return json.load(f)

def normalize_u(u, stats):
    return (u - stats['mean_u']) / stats['std_u']

def denormalize_u(u_norm, stats):
    return (u_norm * stats['std_u']) + stats['mean_u']

def denormalize_c(c_norm, stats):
    """Translates normalized Neural output into Physical units for the solver """
    c_phys = (c_norm * stats['std_c']) + stats['mean_c']
    return np.clip(c_phys, stats['min_c'], stats['max_c'])

# --- 3. CORE PROCESSING LOGIC ---

def get_inverse_and_gt(model, dataset, traj_id, dt_gap, device):
    """Step 1: Get CNO inverse prediction and Ground Truth [cite: 3, 4]"""
    sample_idx = traj_id * 120 + dt_gap - 1 
    time_val, input_tensor, target_c_norm = dataset[sample_idx]
    
    time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_norm = model(input_batch, time_tensor)
    
    # Store normalized versions for Neural Forward Pass
    c_pred_norm_tensor = prediction_norm[:, 0:1] # [1, 1, H, W]
    c_pred_norm_np = c_pred_norm_tensor[0, 0].cpu().numpy()
    c_gt_norm_np = target_c_norm[0].cpu().numpy()
    
    # Extract GT trajectory (already in physical units) 
    i = sample_idx // dataset.multiplier
    h5_idx = i + dataset.start
    u_gt_traj = dataset.reader['u'][h5_idx]

    return c_pred_norm_tensor, c_pred_norm_np, c_gt_norm_np, u_gt_traj

def run_forward_cno_rollout(fwd_model, c_pred_tensor, u0_phys, stats, device):
    """Step 2: Autoregressive Neural Rollout (Batched GPU Optimized)"""
    fwd_model.eval()
    
    # Static Predicted Velocity (Channel 1) - Shape: [Batch, 1, 128, 128]
    c_tensor = c_pred_tensor.to(device)
    batch_size = c_tensor.shape[0]
    
    # dt passed as a spatial map (Channel 2)
    dt_val = 1.0 / stats['n_steps'] 
    
    # Expand t_scalar to match the batch size: [Batch, 1]
    t_scalar = torch.ones((batch_size, 1), dtype=torch.float32, device=device) * dt_val
    t_channel = torch.ones_like(c_tensor) * dt_val 
    
    # Move stats to GPU for fast tensor math
    mean_u_t = torch.tensor(stats['mean_u'], dtype=torch.float32, device=device)
    std_u_t = torch.tensor(stats['std_u'], dtype=torch.float32, device=device)

    # Initialize dynamic wave tensor on GPU
    # u0_phys is [Batch, 128, 128], we need [Batch, 1, 128, 128]
    current_u_tensor = torch.from_numpy(u0_phys).float().to(device).unsqueeze(1)
    
    # Keep track of tensors on GPU, stripping the channel dim -> [Batch, 128, 128]
    preds_phys_tensors = [current_u_tensor.squeeze(1)]

    for _ in range(1, 16):
        # Normalize directly on GPU
        u_norm_tensor = (current_u_tensor - mean_u_t) / std_u_t
        
        # Concatenate ALL 3 CHANNELS: [u, c, dt_map] -> [Batch, 3, 128, 128]
        x_in = torch.cat([u_norm_tensor, c_tensor, t_channel], dim=1)
        
        with torch.no_grad():
            pred_out = fwd_model(x_in, t_scalar)
            
        # Slicing out ONLY the wave field (channel 0)
        u_next_norm = pred_out[:, 0:1]
        
        # Denormalize directly on GPU
        current_u_tensor = (u_next_norm * std_u_t) + mean_u_t
        
        preds_phys_tensors.append(current_u_tensor.squeeze(1))
        
    # Stack all 16 frames: [16, Batch, 128, 128] and move to CPU/NumPy ONCE
    return torch.stack(preds_phys_tensors).cpu().numpy()


def run_numerical_resimulation(c_phys, u0_phys):
    """Step 3: FDTD Numerical Physics Simulation [cite: 1, 399]"""
    NX, NY = 128, 128
    DELTAX, DELTAY = 40, 40 
    DELTAT = 0.002 
    d_steps = 40
    NSTEPS = 15 * d_steps + 1 
    NPOINTS_PML = 10
    f0 = 0.5
    dtype = np.float32

    density = np.ones((NX, NY), dtype=dtype)
    initial_pressures = (u0_phys.astype(dtype), u0_phys.astype(dtype))

    wavefields, _ = seismicCPML2D(
        NX=NX, NY=NY, NSTEPS=NSTEPS, DELTAX=DELTAX, DELTAY=DELTAY, DELTAT=DELTAT,
        NPOINTS_PML=NPOINTS_PML, velocity=c_phys.astype(dtype), density=density,
        initial_pressures=initial_pressures, f0=f0, dtype=dtype, output_wavefields=True
    )
    return wavefields[::d_steps]

def evaluate_trajectory_accuracies(u_gt_batch, u_num_batch, u_cno_batch, save_path):
    """
    Calculates and plots the mean and +/- 1 std of cumulative relative L2 errors.
    Expected input shapes: [Batch, 16, 128, 128]
    """
    eps = 1e-8
    
    def calc_cum_error_stats(pred, true):
        # 1. Sum squared differences and true values over spatial dimensions (X, Y)
        diff_sq = np.sum((pred - true)**2, axis=(2, 3)) # Shape: [Batch, 16]
        true_sq = np.sum(true**2, axis=(2, 3))          # Shape: [Batch, 16]
        
        # 2. Cumulative sum over the time dimension
        cum_diff_sq = np.cumsum(diff_sq, axis=1)        # Shape: [Batch, 16]
        cum_true_sq = np.cumsum(true_sq, axis=1)        # Shape: [Batch, 16]
        
        # 3. Calculate Relative L2 per trajectory [Batch, 16]
        rel_l2 = np.sqrt(cum_diff_sq) / (np.sqrt(cum_true_sq) + eps)
        
        # 4. Calculate Mean and Std across the batch
        return np.mean(rel_l2, axis=0), np.std(rel_l2, axis=0)

    # Calculate statistics for the 3 curves (returns mean, std)
    mean_cno_gt, std_cno_gt = calc_cum_error_stats(u_cno_batch, u_gt_batch)
    mean_num_gt, std_num_gt = calc_cum_error_stats(u_num_batch, u_gt_batch)
    mean_cno_num, std_cno_num = calc_cum_error_stats(u_cno_batch, u_num_batch)

    # Print Report
    print("\n" + "="*50)
    print("               ACCURACY REPORT (Final Step Error)")
    print("="*50)
    print(f"1. FDTD vs Ground Truth : {mean_num_gt[-1]:.4f} ± {std_num_gt[-1]:.4f}  (Physical Baseline)")
    print(f"2. CNO vs Ground Truth  : {mean_cno_gt[-1]:.4f} ± {std_cno_gt[-1]:.4f}  (End-to-End Surrogate)")
    print(f"3. CNO vs FDTD          : {mean_cno_num[-1]:.4f} ± {std_cno_num[-1]:.4f}  (Surrogate Fidelity)")
    print("="*50)

    # Plotting
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
    
# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_gap = 9
    num_test_traj = 120 # Set to 120 to see the real massive speedup

    # Load Inverse Model
    inv_model, inv_dataset = make_test_dataset_from_ckpt(
        model_name="model_0103_new_data", checkpoint_name="epoch=3-step=30720.ckpt", num_traj=num_test_traj
    )
    inv_model.to(device).eval()

    # Load Forward Model via WaveLoader
    fwd_base = f"{PROJECT_ROOT}/results/checkpoints/model_1401"
    fwd_loader = WaveLoader(
        train_file=os.path.join(fwd_base, "training_properties.txt"),
        net_file=os.path.join(fwd_base, "net_architecture.txt"),
        ckpt_path=os.path.join(fwd_base, "epoch=45-step=100096.ckpt"),
        which_name="wave_gauss"
    )
    fwd_model = fwd_loader.load_model()
    
    stats = load_json_stats(inv_dataset.data_path)
    statsfwd = {
        "mean_u": 0.03643418821194553, "std_u": 0.11798792140217046,
        "mean_c": 2631.6621610833695, "std_c": 607.6258083056986,
        "min_c": 1409.754736965702, "max_c": 5419.336611031134,
        "n_steps": 15
    }
    stats.update(statsfwd)

    print(f"Gathering {num_test_traj} samples into a single batch...")
    
    # Lists to hold batched data
    c_pred_tensors = []
    u0_phys_list = []
    c_pred_phys_list = []
    u_gt_traj_list = []

    for traj_id in range(num_test_traj):
        c_tensor, c_pred_norm, _, u_gt_traj = get_inverse_and_gt(inv_model, inv_dataset, traj_id, dt_gap, device)
        c_pred_tensors.append(c_tensor.squeeze(0)) # Remove batch dim temporarily
        u0_phys_list.append(u_gt_traj[0])
        c_pred_phys_list.append(denormalize_c(c_pred_norm, stats))
        u_gt_traj_list.append(u_gt_traj)

    # CREATE BATCHES
    # [Batch, 1, 128, 128]
    c_batch_tensor = torch.stack(c_pred_tensors).to(device) 
    # [Batch, 128, 128]
    u0_batch_phys = np.stack(u0_phys_list)

    # --- WARMUP GPU ---
    print("Warming up GPU...")
    _ = run_forward_cno_rollout(fwd_model, c_batch_tensor[0:1], u0_batch_phys[0:1], stats, device)
    torch.cuda.synchronize()

    # --- 1. TIMING BATCHED NEURAL FORWARD ROLLOUT ---
    print(f"\n--- Running BATCHED Neural Forward prediction (Batch Size = {num_test_traj}) ---")
    torch.cuda.synchronize()
    start_fwd = time.time()
    
    u_fwd_cno_batch = run_forward_cno_rollout(fwd_model, c_batch_tensor, u0_batch_phys, stats, device)
    
    torch.cuda.synchronize()
    fwd_total_time = time.time() - start_fwd
    # Shape of u_fwd_cno_batch: [16 steps, Batch, 128, 128] -> Transpose to [Batch, 16, 128, 128]
    u_fwd_cno_batch = u_fwd_cno_batch.transpose(1, 0, 2, 3) 

    # --- 2. TIMING SEQUENTIAL NUMERICAL RESIMULATION ---
    print("\n--- Running Sequential FDTD Resimulation ---")
    start_num = time.time()
    u_num_resim_list = []
    for traj_id in range(num_test_traj):
        u_num_resim = run_numerical_resimulation(c_pred_phys_list[traj_id], u0_phys_list[traj_id])
        u_num_resim_list.append(u_num_resim)
    num_total_time = time.time() - start_num

    # --- 3. REPORTING ---
    print("\n" + "="*50)
    print("               PERFORMANCE REPORT")
    print("="*50)
    print(f"Total Trajectories Simulated : {num_test_traj}")
    print(f"FDTD Total Time (Sequential) : {num_total_time:.4f}s  (~{num_total_time/num_test_traj:.4f}s per traj)")
    print(f"CNO Total Time (Batched)     : {fwd_total_time:.4f}s  (~{fwd_total_time/num_test_traj:.4f}s per traj)")
    print(f"TOTAL SPEEDUP FACTOR         : {num_total_time / fwd_total_time:.1f}x")
    print("="*50)

    print("\nProcessing Complete. All three versions of u are generated.")

    # --- 4. ACCURACY EVALUATION ---
    # Convert lists to batched numpy arrays: [Batch, 16, 128, 128]
    u_gt_batch = np.stack(u_gt_traj_list)
    u_num_batch = np.stack(u_num_resim_list)
    # u_fwd_cno_batch is already [Batch, 16, 128, 128]

    eval_save_path = f"{PROJECT_ROOT}/accuracy_comparison.pdf"
    evaluate_trajectory_accuracies(u_gt_batch, u_num_batch, u_fwd_cno_batch, eval_save_path)