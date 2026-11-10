import os
import sys
import torch
import time
import numpy as np
import json
from tqdm import tqdm
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
    """Provides raw physical units without artificial range-clipping."""
    return (c_norm * stats['std_c']) + stats['mean_c']

# --- 3. CORE PROCESSING LOGIC ---

def get_inverse_and_gt(model, dataset, traj_id, dt_gap, device):
    """Step 1: Get CNO inverse prediction and Ground Truth"""
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

class CNORollout(torch.nn.Module):
    """
    Implements internal model timing as requested by the supervisor.
    Isolates only the model execution from all other logic.
    """
    def __init__(self, model, steps=15):
        super().__init__()
        self.model = model
        self.steps = steps

    def forward(self, u_norm, c_tensor, t_channel, t_scalar):
        preds = [u_norm]
        curr_u = u_norm
        total_model_math_time = 0.0 # To aggregate the 15 durations
        
        for _ in range(1, self.steps + 1):
            # Pre-processing (Excluded from pure timing)
            x_in = torch.cat([curr_u, c_tensor, t_channel], dim=1)
            
            # --- THE SUPERVISOR SANDWICH ---
            torch.cuda.synchronize() # Clear GPU queue
            t_start = time.time()
            
            pred_out = self.model(x_in, t_scalar)
            
            torch.cuda.synchronize() # Wait for math to finish
            t_end = time.time()
            # --------------------------------------------------------
            
            total_model_math_time += (t_end - t_start)
            
            curr_u = pred_out[:, 0:1]
            preds.append(curr_u)
            
        return torch.stack(preds), total_model_math_time



def measure_forward_pass(rollout_module, c_batch_tensor, u0_phys, stats, device):
    """
    Implements the supervisor's 'sandwich' timing method.
    Isolates the neural execution from all pre/post-processing.
    """
    # PRE-PROCESSING (Not Timed)
    c_tensor = c_batch_tensor.to(device)
    batch_size = c_tensor.shape[0]
    dt_val = 1.0 / stats['n_steps']
    t_scalar = torch.ones((batch_size, 1), device=device) * dt_val
    t_channel = torch.ones_like(c_tensor) * dt_val
    
    mean_u = torch.tensor(stats['mean_u'], device=device)
    std_u = torch.tensor(stats['std_u'], device=device)
    u0_tensor = torch.from_numpy(u0_phys).float().to(device).unsqueeze(1)
    u_norm = (u0_tensor - mean_u) / std_u

    # THE SANDWICH (Timed)
    torch.cuda.synchronize() # Wait for GPU to be ready
    start_time = time.time()
    
    with torch.no_grad():
        # The actual forward pass of the neural surrogate
        all_norm = rollout_module(u_norm, c_tensor, t_channel, t_scalar)
        
    torch.cuda.synchronize() # Wait for GPU to finish
    end_time = time.time()
    
    inference_time = end_time - start_time

    # POST-PROCESSING (Not Timed)
    all_phys = (all_norm * std_u) + mean_u
    res_np = all_phys.squeeze(2).transpose(0, 1).cpu().numpy()
    
    return res_np, inference_time

def run_forward_cno_rollout_compiled(compiled_wrapper, c_pred_tensor, u0_phys, stats, device):
    """
    Executes a single batch rollout. Logic is isolated for timing.
    """
    c_tensor = c_pred_tensor.to(device)
    batch_size = c_tensor.shape[0]
    dt_val = 1.0 / stats['n_steps'] 
    t_scalar = torch.ones((batch_size, 1), device=device) * dt_val
    t_channel = torch.ones_like(c_tensor) * dt_val 
    mean_u = torch.tensor(stats['mean_u'], device=device)
    std_u = torch.tensor(stats['std_u'], device=device)
    
    u0_tensor = torch.from_numpy(u0_phys).float().to(device).unsqueeze(1)
    u_norm = (u0_tensor - mean_u) / std_u

    torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        all_norm = compiled_wrapper(u_norm, c_tensor, t_channel, t_scalar)
    torch.cuda.synchronize()
    inf_time = time.time() - t_start

    all_phys = (all_norm * std_u) + mean_u
    res = all_phys.squeeze(2).transpose(0, 1).cpu().numpy()
    return res, inf_time


def run_numerical_resimulation(c_phys, u0_phys):
    """Step 3: FDTD Numerical Physics Simulation"""
    NX, NY = 128, 128
    DELTAX, DELTAY = 40, 40 
    DELTAT = 0.002 
    d_steps = 40
    NSTEPS = 15 * d_steps + 1 
    NPOINTS_PML = 10
    f0 = 0.5
    dtype = np.float32
    # Check for unphysical velocities that would crash FDTD (Stability check)
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
    # --- HARDWARE OPTIMIZATION ---
    # Essential for RTX 4090 throughput on Euler
    torch.set_float32_matmul_precision('high') 
    torch.backends.cudnn.benchmark = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_samples = 120 
    dt_gap = 9

    # 1. Load Everything
    inv_model, inv_dataset = make_test_dataset_from_ckpt(
        model_name="model_0103_new_data", 
        checkpoint_name="epoch=3-step=30720.ckpt", 
        num_traj=total_samples
    )
    inv_model.to(device).eval()

    fwd_base = os.path.join(PROJECT_ROOT, "fwd_results/checkpoints/model_0903_newgauss")
    fwd_loader = WaveLoader(
        train_file=os.path.join(fwd_base, "training_properties.txt"),
        net_file=os.path.join(fwd_base, "net_architecture.txt"),
        ckpt_path=os.path.join(fwd_base, "epoch=20-step=182784.ckpt"),
        which_name="wave_gauss"
    )
    fwd_model = fwd_loader.load_model()
    stats = load_json_stats(inv_dataset.data_path)
    # --- DIAGNOSTIC: DEVICE CHECK ---
    print("\n[Diagnostic] Verifying Model Device Placement...")
    all_on_gpu = True
    for name, param in fwd_model.named_parameters():
        if not param.is_cuda:
            print(f"  WARNING: Parameter {name} is on CPU!")
            all_on_gpu = False

    # Check for hidden buffers (common in CNO for filters/FFT grids)
    for name, buf in fwd_model.named_buffers():
        if not buf.is_cuda:
            print(f"  WARNING: Buffer {name} is on CPU!")
            all_on_gpu = False

    if all_on_gpu:
        print("  Success: All parameters and buffers are on the GPU.")
    # --------------------------------

    # 2. Setup Compiled Surrogate
    print("\n[Hardware] Compiling Neural Surrogate for RTX 4090...")
    fwd_model = torch.compile(fwd_model) 

    print(f"Activation type used in model: {fwd_model.encoder[0].activation.__class__.__name__}")
    rollout_module = CNORollout(fwd_model, steps=15).to(device)

    # 3. Data Gathering
    print(f"Gathering {total_samples} trajectories...")
    c_pred_tensors, u0_phys_list, c_pred_phys_list, u_gt_traj_list = [], [], [], []
    for traj_id in range(total_samples):
        from fwd_from_inv_prediction import get_inverse_and_gt
        c_tensor, c_pred_norm, _, u_gt_traj = get_inverse_and_gt(inv_model, inv_dataset, traj_id, dt_gap, device)
        c_pred_tensors.append(c_tensor.squeeze(0)) 
        u0_phys_list.append(u_gt_traj[0])
        c_pred_phys_list.append(denormalize_c(c_pred_norm, stats))
        u_gt_traj_list.append(u_gt_traj)

    full_c_batch = torch.stack(c_pred_tensors).to(device)
    full_u0_batch = np.stack(u0_phys_list)
    u_gt_batch = np.stack(u_gt_traj_list)

    # 4. BENCHMARKING PHASE
    print(f"\nPHASE A: NEURAL FORWARD BENCHMARK (Batch Size {total_samples}, N=10 runs)")
    
    # Pre-compute inputs to isolate the "Pure Math"
    mean_u = torch.tensor(stats['mean_u'], device=device)
    std_u = torch.tensor(stats['std_u'], device=device)
    u0_tensor = torch.from_numpy(full_u0_batch).float().to(device).unsqueeze(1)
    u_norm = (u0_tensor - mean_u) / std_u
    
    dt_val = 1.0 / stats['n_steps']
    t_scalar = torch.ones((total_samples, 1), device=device) * dt_val
    t_channel = torch.ones_like(full_c_batch) * dt_val

    total_pure_math_time = 0.0
    num_iterations = 2
    u_fwd_cno_batch = None

    from torch.profiler import profile, record_function, ProfilerActivity
    for i in tqdm(range(num_iterations + 1), desc="Timed Rollouts"):
        # --- DIAGNOSTIC: LAYER-WISE PROFILER ---

        print("\n[Diagnostic] Running Detailed GPU Profiler...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("CNO_Full_Rollout"):
                with torch.no_grad():
                    preds_norm, iteration_math_time = rollout_module(u_norm, full_c_batch, t_channel, t_scalar)
        
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # Skip warmup iteration to exclude initialization overhead
        if i > 0:
            total_pure_math_time += iteration_math_time
        
        # Convert wavefield for accuracy comparison on the final run
        if i == num_iterations:
            all_phys = (preds_norm * std_u) + mean_u
            u_fwd_cno_batch = all_phys.squeeze(2).transpose(0, 1).cpu().numpy()

    avg_pure_math_time = total_pure_math_time / num_iterations

    # 5. FDTD BASELINE
    print("\nPHASE B: FDTD NUMERICAL BASELINE")
    start_num = time.time()
    # Sequential resimulation established in baseline reports
    u_num_resim_list = [run_numerical_resimulation(c_pred_phys_list[i], u0_phys_list[i]) for i in tqdm(range(total_samples), desc="PDE Solver")]
    u_num_batch = np.stack(u_num_resim_list)
    num_total_time = time.time() - start_num

    # 6. FINAL PERFORMANCE REPORT
    time_per_sample_ms = (avg_pure_math_time / total_samples) * 1000 
    
    print("\n" + "="*60)
    print("               FINAL PERFORMANCE REPORT")
    print("="*60)
    print(f"Total Trajectories Simulated  : {total_samples}")
    print(f"Total FDTD Time (Sequential)  : {num_total_time:.4f}s")
    print(f"Avg Pure Forward Math Time    : {avg_pure_math_time:.4f}s")
    print("-" * 60)
    print(f"Average CNO Time per Sample   : {time_per_sample_ms:.4f} ms")
    print(f"FINAL SPEEDUP FACTOR          : {num_total_time / avg_pure_math_time:.2f}x")
    print("="*60)

    # 7. ACCURACY EVALUATION [cite: 70, 102]
    from fwd_from_inv_prediction import evaluate_trajectory_accuracies
    print("\nPHASE C: ACCURACY EVALUATION")
    evaluate_trajectory_accuracies(u_gt_batch, u_num_batch, u_fwd_cno_batch, f"{PROJECT_ROOT}/accuracy_comparison.pdf")
    print(f"\n[Success] Accuracy comparison saved to: {PROJECT_ROOT}/accuracy_comparison.pdf")