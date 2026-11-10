# 
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import copy
import time
import json
import matplotlib.pyplot as plt
from torch.nn import functional as F
from scipy.signal import correlate2d

# --- 1. SETUP PATHS ---
PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
EXTERNAL_CNO_DIR = f"{PROJECT_ROOT}/src/external/CNO/CNO2d_time_dependent_&_foundation_model"
sys.path.append(EXTERNAL_CNO_DIR)
sys.path.append(f"{PROJECT_ROOT}/src/my_models")

from test_and_fine_tune_utils.test_utils import _load_data
from wave_loaders import WaveLoader


def load_json_stats(data_path):
    """Loads normalization statistics from the JSON file."""
    json_path = os.path.join(data_path, "new_normalization_stats.json")
    with open(json_path, 'r') as f:
        return json.load(f)

def _write_in_file(file_name, test_dic, E_l1, E_l2, E_abs, times, pattern, separate_var=False, num_var=0):
    """Writes results to file including mean, median, absolute, and standard deviation metrics."""
    print(f"Writing results to {file_name}")
    with open(file_name, "w") as file:
        max_steps = int(float(test_dic["step_max"]))
        for t in range(len(times)):
            time_val = int(times[t]*max_steps)
            if not separate_var:
                # Relative L1 metrics
                file.write(f"median_l1 {time_val},{np.median(E_l1[t])}\n")
                file.write(f"mean_l1 {time_val},{np.mean(E_l1[t])}\n")
                file.write(f"std_l1 {time_val},{np.std(E_l1[t])}\n")
                
                # Relative L2 metrics
                file.write(f"median_l2 {time_val},{np.median(E_l2[t])}\n")
                file.write(f"mean_l2 {time_val},{np.mean(E_l2[t])}\n")
                file.write(f"std_l2 {time_val},{np.std(E_l2[t])}\n")
                
                # Absolute metrics
                file.write(f"median_abs {time_val},{np.median(E_abs[t])}\n")
                file.write(f"mean_abs {time_val},{np.mean(E_abs[t])}\n")
            else:
                for i in range(num_var):
                    # Relative L1 metrics per variable
                    file.write(f"median_l1 {time_val}_{i},{np.median(E_l1[i,t])}\n")
                    file.write(f"mean_l1 {time_val}_{i},{np.mean(E_l1[i,t])}\n")
                    file.write(f"std_l1 {time_val}_{i},{np.std(E_l1[i,t])}\n")
                    
                    # Relative L2 metrics per variable
                    file.write(f"median_l2 {time_val}_{i},{np.median(E_l2[i,t])}\n")
                    file.write(f"mean_l2 {time_val}_{i},{np.mean(E_l2[i,t])}\n")
                    file.write(f"std_l2 {time_val}_{i},{np.std(E_l2[i,t])}\n")
                    
                    # Absolute metrics per variable
                    file.write(f"median_abs {time_val}_{i},{np.median(E_abs[i,t])}\n")
                    file.write(f"mean_abs {time_val}_{i},{np.mean(E_abs[i,t])}\n")


def plot_comparative_energy(energy_s, energy_gt, save_path):
    """
    Plots Surrogate vs Ground Truth energy stability for report comparison.
    energy_s, energy_gt shape: [Time_Steps, Num_Samples]
    """
    # 1. Normalize both by their own initial energy at t=0
    rel_s = energy_s / (energy_s[0, :] + 1e-10)
    rel_gt = energy_gt / (energy_gt[0, :] + 1e-10)
    
    # 2. Compute statistics across the 120 trajectories
    m_s, s_s = np.mean(rel_s, axis=1).squeeze(), np.std(rel_s, axis=1).squeeze()
    m_gt, s_gt = np.mean(rel_gt, axis=1).squeeze(), np.std(rel_gt, axis=1).squeeze()
    steps = np.arange(len(m_s))

    plt.figure(figsize=(10, 6))
    
    # Plot Ground Truth (The "Physical Target")
    plt.plot(steps, m_gt, label='FDTD Ground Truth', color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.fill_between(steps, m_gt - s_gt, m_gt + s_gt, color='black', alpha=0.1)
    
    # Plot Surrogate (The CNO Prediction)
    plt.plot(steps, m_s, label='CNO Surrogate', color='#1f77b4', linewidth=2)
    plt.fill_between(steps, m_s - s_s, m_s + s_s, color='#1f77b4', alpha=0.2)

    # Scientific Formatting
    plt.axhline(y=1.0, color='red', linestyle=':', label='Ideal Hamiltonian')
    plt.xlabel("Rollout Step (Autoregressive)")
    plt.ylabel("Relative Total Energy ($E_t / E_0$)")
    plt.grid(True, which='both', alpha=0.3, linestyle=':')
    plt.legend(loc='lower left', frameon=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visuals] Comparative energy plot saved to {save_path}")

def compute_energy(u_curr, u_prev, c_phys, dx=20.0, dt_neural=0.08):
    """
    Approximates the physical Hamiltonian: E = 0.5 * integral( (1/c^2)u_t^2 + |grad u|^2 )
    
    u_curr, u_prev: [Batch, 1, H, W] (Physical units)
    c_phys: [Batch, 1, H, W] (m/s)
    """
    # 1. Kinetic Energy: 0.5 * (1/c^2) * (du/dt)^2
    # dt_neural (0.08) is the jump the model is responsible for.
    u_t = (u_curr - u_prev) / dt_neural
    ke_density = 0.5 * (u_t / (c_phys + 1e-8))**2
    ke = torch.mean(ke_density, dim=(-3, -2, -1))
    
    # 2. Potential Energy: 0.5 * |grad u|^2
    # Scaling by dx (20.0) ensures the gradient is in physical units (Pressure/meter)
    grad_x = ((u_curr[:, :, 1:, :] - u_curr[:, :, :-1, :]) / dx)**2
    grad_y = ((u_curr[:, :, :, 1:] - u_curr[:, :, :, :-1]) / dx)**2
    pe_density = 0.5 * (torch.mean(grad_x, dim=(-3, -2, -1)) + torch.mean(grad_y, dim=(-3, -2, -1)))
    
    return ke + pe_density

def plot_energy_decay(energy_matrix, save_path):
    """
    Plots mean energy loss over time with a 1-std deviation shaded area.
    energy_matrix shape: [Time_Steps (T+1), Num_Samples]
    """
    # Normalize energy by initial energy (t=0) to see relative loss/gain
    initial_energy = energy_matrix[0, :]
    relative_energy = energy_matrix / (initial_energy + 1e-10)
    
    mean_energy = np.mean(relative_energy, axis=1).squeeze()
    std_energy = np.std(relative_energy, axis=1).squeeze()
    time_steps = np.arange(len(mean_energy))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, mean_energy, label='Mean Relative Energy', color='blue', linewidth=2)
    plt.fill_between(time_steps, 
                     mean_energy - std_energy, 
                     mean_energy + std_energy, 
                     color='blue', alpha=0.2, label='$\pm 1\sigma$ deviation')
    
    plt.axhline(y=1.0, color='red', linestyle='--', label='Ideal Conservation')
    plt.title("Physical Fidelity: Energy Stability (15-step Rollout)")
    plt.xlabel("Rollout Step")
    plt.ylabel("Relative Energy ($E_t / E_0$)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(save_path); plt.close()
    print(f"[Visuals] Energy plot saved to {save_path}")


def plot_energy_consistency_ratio(energy_s, energy_gt, save_path):
    """
    Plots the ratio of Surrogate Energy to Ground Truth Energy.
    Ideally, this ratio should stay near 1.0.
    
    energy_s: [Time_Steps, Num_Samples]
    energy_gt: [Time_Steps, Num_Samples]
    """
    # 1. Calculate the element-wise ratio
    # Adding epsilon to avoid division by zero if wave leaves domain entirely
    ratio = (energy_s + 1e-10) / (energy_gt + 1e-10)
    
    # 2. Statistical aggregation across the 120 trajectories
    mean_ratio = np.mean(ratio, axis=1).squeeze()
    std_ratio = np.std(ratio, axis=1).squeeze()
    steps = np.arange(len(mean_ratio))

    plt.figure(figsize=(10, 6))
    
    # Plot the consistency line (Central Mean)
    plt.plot(steps, mean_ratio, color='purple', linewidth=2.5, label='Consistency Ratio ($E_{CNO} / E_{0}$)')
    
    # Plot the Shaded Variance Region (+/- 1 Std Dev)
    plt.fill_between(steps, 
                     mean_ratio - std_ratio, 
                     mean_ratio + std_ratio, 
                     color='purple', alpha=0.15, label='$\pm 1\sigma$ Variation')

    # Reference Line at 1.0 (Perfect Physical Replication)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=0.5, label='Physical Parity')
    
    # Report formatting: No title, minimal white space
    plt.xlabel("Rollout Step")
    plt.ylabel("Energy Ratio")
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(loc='lower left', frameon=True)
    
    # Set Y-axis to focus on the 1.0 region (e.g., 0.8 to 1.1) 
    # to show the fine-grained dissipation
    # plt.ylim(0.5, 1.2)
    
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Visuals] Energy consistency ratio plot saved to {save_path}")

def compute_phase_shift(u_pred, u_gt):
    """
    Calculates spatial lag (pixel distance) via 2D cross-correlation peak detection.
    """
    pred = u_pred.squeeze().cpu().numpy()
    gt = u_gt.squeeze().cpu().numpy()
    
    # Normalize to focus on wavefront shape matching
    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
    gt = (gt - gt.mean()) / (gt.std() + 1e-8)
    
    corr = correlate2d(pred, gt, mode='same')
    y_mid, x_mid = np.array(corr.shape) // 2
    y_peak, x_peak = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Euclidean distance of the lag from the center (zero-shift)
    shift = np.sqrt((y_peak - y_mid)**2 + (x_peak - x_mid)**2)
    return shift

def plot_phase_drift(phase_matrix, save_path):
    """
    Plots average pixel shift over rollout with 1-std shaded region.
    phase_matrix shape: [Time_Steps (T+1), Num_Samples]
    """
    mean_shift = np.mean(phase_matrix, axis=1).squeeze()
    std_shift = np.std(phase_matrix, axis=1).squeeze()
    steps = np.arange(len(mean_shift))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_shift, color='forestgreen', marker='o', label='Mean Phase Shift (pixels)')
    plt.fill_between(steps, mean_shift - std_shift, mean_shift + std_shift, 
                     color='forestgreen', alpha=0.2, label='$\pm 1\sigma$ deviation')
    
    plt.title("Physical Fidelity: Temporal Phase Drift")
    plt.xlabel("Rollout Step")
    plt.ylabel("Spatial Lag (Pixels)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(save_path); plt.close()
    print(f"[Visuals] Phase drift plot saved to {save_path}")

def _test_pattern(model, testing_loader, batch_size=10, time_steps=15, dt=1, max_steps=15.0,
                  pattern=[1]*15, device='cuda', folder="", separate_var=False,
                  separate_dim=[], is_masked=None, steady=False, is_airfoil=False):
    
    print(f"Testing Pattern: {pattern} | Full Comparative Physics (Scaled) Enabled")
    delta = dt/max_steps
    eps = 1e-10
    T = []
    
    # History storage for consistency analysis
    energy_history_s = []
    energy_history_gt = []
    phase_history = []
    
    # Error metrics storage
    e_shape = (len(separate_dim), len(pattern), 0) if separate_var else (len(pattern), 0)
    E_l1, E_l2, E_abs = np.zeros(e_shape), np.zeros(e_shape), np.zeros(e_shape)

    # Physical constants for Hamiltonian density calculation
    DX_PHYS = 20.0
    DT_NEURAL = 40 * 0.002 # 0.08s

    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(testing_loader):
            time_batch_, input_batch_, output_batch_ = batch[:3]
            input_batch = input_batch_.to(device)
            output_batch = output_batch_.to(device)
            t_batch = time_batch_.to(device)
            
            # --- 1. INITIALIZATION ---
            inp = input_batch[0::time_steps] 
            stats = model.loader_dictionary
            
            # Prepare physical fields for scaling
            c_phys = (inp[:, 1:2, :, :] * stats['std_c']) + stats['mean_c']
            u_prev_s_phys = (inp[:, 0:1, :, :] * stats['std_u']) + stats['mean_u']
            u_prev_gt_phys = u_prev_s_phys.clone()
            
            # t=0 Baseline
            e0 = compute_energy(u_prev_s_phys, u_prev_s_phys, c_phys, dx=DX_PHYS, dt_neural=DT_NEURAL)
            batch_energy_s, batch_energy_gt = [e0], [e0]
            batch_phases = [torch.zeros(inp.shape[0])] 

            curr_b_size = inp.shape[0]
            curr_e_shape = (len(separate_dim), len(pattern), curr_b_size) if separate_var else (len(pattern), curr_b_size)
            E_curr_l1, E_curr_l2, E_curr_abs = np.zeros(curr_e_shape), np.zeros(curr_e_shape), np.zeros(curr_e_shape)

            acc_jump = 0
            for i, jump in enumerate(pattern):
                acc_jump += jump
                if step == 0: T.append(acc_jump * delta)
                
                # --- 2. FORWARD PASS ---
                t_val = torch.zeros_like(t_batch[0::time_steps]) + jump * delta
                pred_norm = model(inp, t_val)
                out_norm = output_batch[acc_jump - 1::time_steps]

                # --- 3. PHYSICAL TRANSLATION ---
                u_s_phys = (pred_norm[:, 0:1] * stats['std_u']) + stats['mean_u']
                u_gt_phys = (out_norm[:, 0:1] * stats['std_u']) + stats['mean_u']

                # # --- 4. ENERGY TRACKING ---
                # e_s = compute_energy(u_s_phys, u_prev_s_phys, c_phys, dx=DX_PHYS, dt_neural=DT_NEURAL)
                # e_gt = compute_energy(u_gt_phys, u_prev_gt_phys, c_phys, dx=DX_PHYS, dt_neural=DT_NEURAL)
                
                # batch_energy_s.append(e_s)
                # batch_energy_gt.append(e_gt)

                # Update physical previous states for next derivative
                u_prev_s_phys, u_prev_gt_phys = u_s_phys.clone(), u_gt_phys.clone()

                # # Phase Shift Calculation
                # shifts = [compute_phase_shift(pred_norm[:, 0:1][b], out_norm[:, 0:1][b]) for b in range(curr_b_size)]
                # batch_phases.append(torch.tensor(shifts))

                # --- 5. NUMERICAL ERROR CALCULATION ---
                if not separate_var:
                    E_curr_l1[i] = ((torch.mean(abs(pred_norm - out_norm), (-3, -2, -1)) / (torch.mean(abs(out_norm), (-3, -2, -1)) + eps)) * 100).cpu().numpy()
                    E_curr_l2[i] = ((torch.sqrt(torch.sum((pred_norm - out_norm)**2, (-3, -2, -1))) / (torch.sqrt(torch.sum(out_norm**2, (-3, -2, -1))) + eps)) * 100).cpu().numpy()
                    E_curr_abs[i] = torch.mean(abs(pred_norm - out_norm), (-3, -2, -1)).cpu().numpy()
                else:
                    diff = [0]
                    for val in separate_dim: diff.append(diff[-1] + val)
                    for j in range(len(separate_dim)):
                        p_v, o_v = pred_norm[:, diff[j]:diff[j+1]], out_norm[:, diff[j]:diff[j+1]]
                        E_curr_l1[j, i] = ((torch.mean(abs(p_v - o_v), (-3, -2, -1)) / (torch.mean(abs(o_v), (-3, -2, -1)) + eps)) * 100).cpu().numpy()
                        E_curr_l2[j, i] = ((torch.sqrt(torch.sum((p_v - o_v)**2, (-3, -2, -1))) / (torch.sqrt(torch.sum(o_v**2, (-3, -2, -1))) + eps)) * 100).cpu().numpy()
                        E_curr_abs[j, i] = torch.mean(abs(p_v - o_v), (-3, -2, -1)).cpu().numpy()

                # Autoregressive Feedback
                inp[:, 0:1] = pred_norm[:, 0:1]

            # # Collect Batch Results
            # energy_history_s.append(torch.stack(batch_energy_s).cpu().numpy())
            # energy_history_gt.append(torch.stack(batch_energy_gt).cpu().numpy())
            # phase_history.append(torch.stack(batch_phases).cpu().numpy())
            
            E_l1 = np.concatenate((E_l1, E_curr_l1), axis=-1)
            E_l2 = np.concatenate((E_l2, E_curr_l2), axis=-1)
            E_abs = np.concatenate((E_abs, E_curr_abs), axis=-1)

    # # --- 6. AGGREGATED REPORT GENERATION ---
    # full_e_s = np.concatenate(energy_history_s, axis=1)
    # full_e_gt = np.concatenate(energy_history_gt, axis=1)
    
    # # A: Comparative Energy Plot (Decay Profiles)
    # plot_comparative_energy(full_e_s, full_e_gt, os.path.join(folder, "energy_decay_comparison.pdf"))
    
    # # B: Consistency Ratio Plot (The "Raonic Diagnostic")
    # plot_energy_consistency_ratio(full_e_s, full_e_gt, os.path.join(folder, "energy_consistency_ratio.pdf"))
    
    # # C: Phase Drift Plot
    # plot_phase_drift(np.concatenate(phase_history, axis=1), os.path.join(folder, "phase_drift.pdf"))
    
    return T, E_l1, E_l2, E_abs

def _evaluate_models(folders, which_example="wave_gauss", cluster=True, in_dist=True,
                     patterns=[[1]], folder_pattern="errors", steps=15, separate_var=False, 
                     fine_tuned=False, training_folder=None, write_in_file=True):
    
    base_folder = "/cluster/home/svisnjic/Thalassa/fwd_results/checkpoints"
    ckpt_file = "epoch=20-step=182784.ckpt"
    model_dirs = [(base_folder, ckpt_file)]
    sufix = "in_dist_patterns.txt" if in_dist else "out_dist_patterns.txt"
    
    for folder, _model_file in model_dirs:
        _folder = folder if not fine_tuned and training_folder is None else training_folder
        m_dir = "model_0903_newgauss"
        
        train_file = os.path.join(_folder, m_dir, "training_properties.txt")
        net_file   = os.path.join(_folder, m_dir, "net_architecture.txt")
        ckpt_path  = os.path.join(folder, m_dir, _model_file)
        
        loader = WaveLoader(train_file, net_file, ckpt_path, which_name=which_example)
        
        # --- FIX: INJECT STATS BEFORE LOADING MODEL ---
        loader._ensure_config()  # Pre-load configuration from text files
        data_path = os.path.join(PROJECT_ROOT, "data")
        stats = load_json_stats(data_path)
        loader.loader_dict.update(stats) # Merges mean_c, std_c, etc. into the dictionary
        
        try: 
            # Now the model will inherit the updated dictionary
            model = loader.load_model()
        except: 
            continue
        # ----------------------------------------------

        # Override dict to access full 120-trajectory test set
        loader.loader_dict["num_test"] = 120 
        loader.loader_dict["num_trajectories"] = 120
            
        bs = 32 if 32 % steps == 0 else (32 // steps) * steps
        if bs == 0: bs = steps
        
        test_data_loader = _load_data(which=which_example, loader_dict=loader.loader_dict, 
                                     batch_size=bs, which_loader="test", fix_input_to_time_step=0) 
        
        output_dir = os.path.join(folder, folder_pattern)
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        
        for pattern in patterns:
            T, E_l1, E_l2, E_abs = _test_pattern(model=model, testing_loader=test_data_loader, 
                                                dt=loader.loader_dict["dt"], time_steps=loader.loader_dict["time_steps"], 
                                                max_steps=loader.loader_dict["step_max"], pattern=pattern, folder=output_dir,
                                                device=loader.device, batch_size=bs, separate_var=separate_var, 
                                                separate_dim=loader.loader_dict.get("separate_dim", []))
            
            p_str = "".join([str(p) + "_" for p in pattern])
            error_file = os.path.join(output_dir, p_str + sufix)
            if write_in_file:
                _write_in_file(error_file, loader.loader_dict, E_l1, E_l2, E_abs, T, pattern, 
                              separate_var=separate_var, num_var=len(loader.loader_dict.get("separate_dim", [])))
            print(f"Results for pattern {pattern} saved to {error_file}")

if __name__ == "__main__":
    base_checkpoints_dir = "/cluster/home/svisnjic/Thalassa/fwd_results/checkpoints"
    target_example = "wave_gauss"
    
    # Pattern [1]*15 provides the most high-resolution temporal diagnostic
    diagnostic_patterns = [[1] * 15]

    print("\n" + "="*60)
    print("      CLOSED-LOOP FORWARD MODEL: FULL PHYSICS DIAGNOSTIC      ")
    print("="*60)
    print(f"Targeting: 120 Test Trajectories")
    print(f"Metrics: Rel L1, Rel L2, Phase Drift (px), Energy Stability")
    print("="*60 + "\n")

    _evaluate_models(
        folders=base_checkpoints_dir,
        which_example=target_example,
        cluster=True,
        in_dist=True,
        patterns=diagnostic_patterns,
        folder_pattern="full_physics_diagnostic",
        steps=15, 
        separate_var=True,
        write_in_file=True
    )

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print(f"Metrics folder: full_physics_diagnostic")
    print("Plots generated: energy_stability.pdf, phase_drift.pdf")
    print("="*60)