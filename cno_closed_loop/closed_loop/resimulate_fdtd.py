import os
import sys
import torch
import time
import numpy as np
import json
import matplotlib.pyplot as plt

# Paths & Imports
PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
MYMODELS_DIR = f"{PROJECT_ROOT}/src/my_models"
sys.path.append(MYMODELS_DIR)

# Assuming these are available from your previous modularization
from inverse_problem.inv_testing import make_test_dataset_from_ckpt


sys.path.append(f"{PROJECT_ROOT}/src/external/wave_simulation")
from seismic_CPML_2D_pressure_second_order2 import seismicCPML2D



def get_cno_prediction(model, dataset, traj_id, dt_gap):
    """
    Consolidates logic to get CNO prediction and the Ground Truth trajectory.
    """

    sample_idx = traj_id * 120 + dt_gap - 1 
    time_val, input_tensor, target_c_norm = dataset[sample_idx]
    
    time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_norm = model(input_batch, time_tensor)
    
    c_pred_norm = prediction_norm[0, 0].cpu().numpy()
    c_gt_norm = target_c_norm[0].cpu().numpy()
    
    # Extract the full 16 frames directly from the reader. 
    # This data is ALREADY IN PHYSICAL UNITS (unnormalized).
    i = sample_idx // dataset.multiplier
    h5_idx = i + dataset.start
    u_gt_traj_phys = dataset.reader['u'][h5_idx]

    return c_pred_norm, c_gt_norm, u_gt_traj_phys, dataset.data_path


def denormalize_velocity(c_norm, data_path):
    """
    Translates normalized Neural output back into Physical units for the solver.
    (We no longer need to pass 'u' here, as we pull it physically from the start).
    """
    json_path = os.path.join(data_path, "new_normalization_stats.json")
    with open(json_path, 'r') as f:
        stats = json.load(f)

    c_phys = (c_norm * stats['std_c']) + stats['mean_c']
    c_phys = np.clip(c_phys, stats['min_c'], stats['max_c'])

    return c_phys


def run_physics_validation(c_phys, u0_phys):
    """
    Step 4: Feeds the physical velocity and initial wavefield back into the numerical solver.
    """

    # 1. Match Physical Parameters to the 128x128 CNO space
    NX, NY = 128, 128
    DELTAX, DELTAY = 40, 40 # Scaled up from 20 because resolution is halved (256 -> 128)
    DELTAT = 0.002 # Same dt as generation
    d_steps = 40
    n_steps = 15
    NSTEPS = n_steps * d_steps + 1 # 601 steps total
    NPOINTS_PML = 10
    f0 = 0.5
    dtype = np.float32

    # 2. Setup Solver Inputs
    density = np.ones((NX, NY), dtype=dtype)
    # The solver expects (past_pressure, present_pressure).
    # Just like your generation script, we pass u0 for both to start from rest.
    initial_pressures = (u0_phys.astype(dtype), u0_phys.astype(dtype))

    # 3. Run the Physics Simulation
    print("Running seismicCPML2D numerical resimulation...")
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

    # 4. Time-Domain Downsampling
    # Extract the 15 frames exactly as the generation script did
    u_resimulated_trajectory = wavefields[::d_steps]

    return u_resimulated_trajectory

def plot_trajectories_side_by_side(u_gt_traj, u_resim_traj, save_path):
    """
    Plots the Ground Truth and Resimulated trajectories back-to-back.
    """
    # Use global min/max across BOTH trajectories for a fair visual comparison
    u_min = min(np.min(u_gt_traj), np.min(u_resim_traj))
    u_max = max(np.max(u_gt_traj), np.max(u_resim_traj))
    
    fig, axes = plt.subplots(8, 4, figsize=(20, 32))
    
    # Plot Ground Truth (Rows 0-3)
    for i in range(16):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        im = ax.imshow(u_gt_traj[i], vmin=u_min, vmax=u_max, cmap='jet')
        ax.set_title(f"GT u(t={i})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Plot Resimulation (Rows 4-7)
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
    """
    Calculates and plots step-by-step, cumulative, and total Relative L2 errors.
    """
    eps = 1e-8
    step_errors = []
    cum_num, cum_den = 0.0, 0.0
    cum_errors = []

    print("\n--- Error Report ---")
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
    print(f"TOTAL RELATIVE L2 ERROR: {total_error:.4f}\n")
    
    # Plotting
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

    dt = 9
    num_test_trajectories = 120
    total_errors = []
    all_cum_errors = []

    print(f"Starting closed-loop evaluation for {num_test_trajectories} trajectories at dt={dt}...")
    start_time = time.time()

    test_num_traj = 120
    model, dataset = make_test_dataset_from_ckpt(model_name="model_0103_new_data", checkpoint_name="epoch=3-step=30720.ckpt", num_traj=test_num_traj)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    for traj in range(num_test_trajectories):
        print("--- STEP 1 & 2: CNO Inference ---")
        c_pred_norm, c_gt_norm, u_gt_traj_phys, dat_path = get_cno_prediction(
            model=model,
            dataset=dataset,
            traj_id=traj,
            dt_gap=dt
        )
        
        print("--- STEP 3: Denormalization to Physics ---")
        c_pred_phys = denormalize_velocity(c_norm=c_pred_norm, data_path=dat_path)
        
        # We take the exact physical starting frame from the raw ground truth trajectory
        u0_phys = u_gt_traj_phys[0]
        
        print("--- STEP 4: Numerical Resimulation ---")
        u_resim_trajectory = run_physics_validation(c_phys=c_pred_phys, u0_phys=u0_phys)
        
        print(f"\nSuccess! Recovered trajectory shape: {u_resim_trajectory.shape}")

        print("--- STEP 5: Plotting Results ---")
        
        # De-normalize GT velocity for accurate tracking
        json_path = os.path.join(dat_path, "new_normalization_stats.json")
        with open(json_path, 'r') as f:
            stats = json.load(f)
        c_gt_phys = (c_gt_norm * stats['std_c']) + stats['mean_c']
        
        out_dir = f"{PROJECT_ROOT}/inv_results/testing_visuals/closed_loop"
        os.makedirs(out_dir, exist_ok=True)
        
        # 3. Calculate Step-by-Step Cumulative Error (Fast NumPy Vectorization)
        eps = 1e-8
        
        # Sum the squares spatially (over X and Y) for each time step -> Shape: (16,)
        diff_sq_per_step = np.sum((u_resim_trajectory - u_gt_traj_phys)**2, axis=(1, 2))
        gt_sq_per_step = np.sum(u_gt_traj_phys**2, axis=(1, 2))
        
        # Cumulative sum over time -> Shape: (16,)
        cum_diff_sq = np.cumsum(diff_sq_per_step)
        cum_gt_sq = np.cumsum(gt_sq_per_step)
        
        # Final cumulative error curve for THIS trajectory
        traj_cum_errors = np.sqrt(cum_diff_sq) / (np.sqrt(cum_gt_sq) + eps)
        
        all_cum_errors.append(traj_cum_errors)

        # Print the FINAL total error for this specific trajectory to track progress
        print(f"Traj {traj+1:03d}/{num_test_trajectories} | Final L2 Error: {traj_cum_errors[-1]:.4f}")


        # comparison_save_file = f"{out_dir}/resimulation_vs_gt_traj{traj}_dt{dt}.pdf"
        
        # plot_trajectories_side_by_side(
        #     u_gt_traj=u_gt_traj_phys,
        #     u_resim_traj=u_resim_trajectory, 
        #     save_path=comparison_save_file
        # )

        # # Calculate and plot errors
        # error_save_file = f"{out_dir}/resimulation_errors_traj{traj}_dt{dt}.pdf"
        # calculate_and_plot_errors(
        #     u_gt_traj=u_gt_traj_phys, 
        #     u_resim_traj=u_resim_trajectory, 
        #     save_path=error_save_file
        # )
    
    all_cum_errors = np.array(all_cum_errors) # Shape: (120, 16)
    
    mean_cum_errors = np.mean(all_cum_errors, axis=0) # Shape: (16,)
    std_cum_errors = np.std(all_cum_errors, axis=0)   # Shape: (16,)
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n========================================")
    print(f"EVALUATION COMPLETE ({elapsed:.1f} minutes)")
    print("========================================")
    print(f"Average Final Cumulative Error: {mean_cum_errors[-1]:.4f} ± {std_cum_errors[-1]:.4f}")
    print("========================================")

    # --- 5. PLOT THE AVERAGE CURVE ---
    out_dir = f"{PROJECT_ROOT}/inv_results/testing_visuals/closed_loop"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    steps = np.arange(16)
    
    # Plot the mean line
    plt.plot(steps, mean_cum_errors, label='Mean Cumulative Error', color='#ff7f0e', linewidth=2, marker='o')
    
    # Add a shaded region for the standard deviation (shows variance across the 120 samples)
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
    print(f"Saved average error plot to: {plot_path}")