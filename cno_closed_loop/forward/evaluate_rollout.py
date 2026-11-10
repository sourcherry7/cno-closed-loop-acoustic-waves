"""Autoregressive rollout evaluation of the forward CNO on the 120 test trajectories.

Rolls the trained forward model out from u(t=0) with the true velocity field and
writes relative L1/L2 and absolute errors per step (the report's forward results).
Optionally (PHYSICS_DIAGNOSTICS) it also tracks the discrete wave energy and the
phase drift of the predicted wavefronts.
Run from the repository root:  python -m cno_closed_loop.forward.evaluate_rollout
"""

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

from cno_closed_loop import paths
from test_and_fine_tune_utils.test_utils import _load_data
from cno_closed_loop.forward.model_loader import WaveLoader


def load_json_stats(data_path):
    """Loads normalization statistics from the JSON file."""
    json_path = os.path.join(data_path, "new_normalization_stats.json")
    with open(json_path, 'r') as f:
        return json.load(f)

def _write_in_file(file_name, test_dic, E_l1, E_l2, E_abs, times, pattern, separate_var=False, num_var=0):
    """Write median/mean/std of the relative L1 and L2 errors and the absolute error per step."""
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
    """Relative energy E_t / E_0 of the CNO rollout and of the ground truth.

    energy_s, energy_gt: [time_steps, num_samples]
    """
    # Normalize each trajectory by its own energy at t = 0
    rel_s = energy_s / (energy_s[0, :] + 1e-10)
    rel_gt = energy_gt / (energy_gt[0, :] + 1e-10)
    
    m_s, s_s = np.mean(rel_s, axis=1).squeeze(), np.std(rel_s, axis=1).squeeze()
    m_gt, s_gt = np.mean(rel_gt, axis=1).squeeze(), np.std(rel_gt, axis=1).squeeze()
    steps = np.arange(len(m_s))

    plt.figure(figsize=(10, 6))
    
    plt.plot(steps, m_gt, label='FDTD Ground Truth', color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.fill_between(steps, m_gt - s_gt, m_gt + s_gt, color='black', alpha=0.1)
    
    plt.plot(steps, m_s, label='CNO Surrogate', color='#1f77b4', linewidth=2)
    plt.fill_between(steps, m_s - s_s, m_s + s_s, color='#1f77b4', alpha=0.2)

    plt.axhline(y=1.0, color='red', linestyle=':', label='Energy conservation')
    plt.xlabel("Rollout Step (Autoregressive)")
    plt.ylabel("Relative Total Energy ($E_t / E_0$)")
    plt.grid(True, which='both', alpha=0.3, linestyle=':')
    plt.legend(loc='lower left', frameon=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def compute_energy(u_curr, u_prev, c_phys, dx=20.0, dt_neural=0.08):
    """Discrete wave energy E = 0.5 * mean( (u_t / c)^2 + |grad u|^2 ).

    u_curr, u_prev: [batch, 1, H, W] wavefields in physical units
    c_phys:         [batch, 1, H, W] velocity in m/s
    """
    # Kinetic term, with u_t from the difference over one model step (dt_neural)
    u_t = (u_curr - u_prev) / dt_neural
    ke_density = 0.5 * (u_t / (c_phys + 1e-8))**2
    ke = torch.mean(ke_density, dim=(-3, -2, -1))
    
    # Potential term, with the gradient in physical units (dx in m)
    grad_x = ((u_curr[:, :, 1:, :] - u_curr[:, :, :-1, :]) / dx)**2
    grad_y = ((u_curr[:, :, :, 1:] - u_curr[:, :, :, :-1]) / dx)**2
    pe_density = 0.5 * (torch.mean(grad_x, dim=(-3, -2, -1)) + torch.mean(grad_y, dim=(-3, -2, -1)))
    
    return ke + pe_density

def plot_energy_consistency_ratio(energy_s, energy_gt, save_path):
    """Ratio of CNO energy to ground-truth energy per step (ideally 1).

    energy_s, energy_gt: [time_steps, num_samples]
    """
    # Epsilon avoids division by zero if the wave has left the domain
    ratio = (energy_s + 1e-10) / (energy_gt + 1e-10)
    
    mean_ratio = np.mean(ratio, axis=1).squeeze()
    std_ratio = np.std(ratio, axis=1).squeeze()
    steps = np.arange(len(mean_ratio))

    plt.figure(figsize=(10, 6))
    
    plt.plot(steps, mean_ratio, color='purple', linewidth=2.5, label=r'$E_{CNO} / E_{GT}$')
    plt.fill_between(steps, 
                     mean_ratio - std_ratio, 
                     mean_ratio + std_ratio, 
                     color='purple', alpha=0.15, label=r'$\pm 1\sigma$')
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=0.5, label='Ratio = 1')
    
    plt.xlabel("Rollout Step")
    plt.ylabel("Energy Ratio")
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def compute_phase_shift(u_pred, u_gt):
    """Spatial lag in pixels between prediction and ground truth (peak of the 2D cross-correlation)."""
    pred = u_pred.squeeze().cpu().numpy()
    gt = u_gt.squeeze().cpu().numpy()
    
    # Standardize so that only the wavefront shape matters
    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
    gt = (gt - gt.mean()) / (gt.std() + 1e-8)
    
    corr = correlate2d(pred, gt, mode='same')
    y_mid, x_mid = np.array(corr.shape) // 2
    y_peak, x_peak = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Distance of the correlation peak from the zero-shift position
    shift = np.sqrt((y_peak - y_mid)**2 + (x_peak - x_mid)**2)
    return shift

def plot_phase_drift(phase_matrix, save_path):
    """Mean phase drift (pixels) per rollout step with a 1-sigma band; phase_matrix: [T+1, num_samples]."""
    mean_shift = np.mean(phase_matrix, axis=1).squeeze()
    std_shift = np.std(phase_matrix, axis=1).squeeze()
    steps = np.arange(len(mean_shift))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_shift, color='forestgreen', marker='o', label='Mean Phase Shift (pixels)')
    plt.fill_between(steps, mean_shift - std_shift, mean_shift + std_shift, 
                     color='forestgreen', alpha=0.2, label=r'$\pm 1\sigma$')
    
    plt.title("Phase drift during the rollout")
    plt.xlabel("Rollout Step")
    plt.ylabel("Spatial Lag (Pixels)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(save_path); plt.close()
    print(f"Saved {save_path}")

def _test_pattern(model, testing_loader, batch_size=10, time_steps=15, dt=1, max_steps=15.0,
                  pattern=[1]*15, device='cuda', folder="", separate_var=False,
                  separate_dim=[], is_masked=None, steady=False, is_airfoil=False,
                  physics_diagnostics=False):
    """Autoregressive rollout following `pattern`; returns times and L1/L2/abs errors per step.

    With physics_diagnostics=True it also tracks the energy of the CNO and ground-truth
    wavefields and the phase drift, and saves the corresponding plots to `folder`.
    """
    print(f"Testing pattern {pattern} (physics diagnostics: {'on' if physics_diagnostics else 'off'})")
    delta = dt/max_steps
    eps = 1e-10
    T = []
    
    # Per-batch histories for the optional physics diagnostics
    energy_history_s = []
    energy_history_gt = []
    phase_history = []
    
    e_shape = (len(separate_dim), len(pattern), 0) if separate_var else (len(pattern), 0)
    E_l1, E_l2, E_abs = np.zeros(e_shape), np.zeros(e_shape), np.zeros(e_shape)

    # Constants for the energy diagnostic: grid spacing (m) and time per model step (s)
    DX_PHYS = 20.0
    DT_NEURAL = 40 * 0.002  # 40 solver steps of 2 ms

    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(testing_loader):
            time_batch_, input_batch_, output_batch_ = batch[:3]
            input_batch = input_batch_.to(device)
            output_batch = output_batch_.to(device)
            t_batch = time_batch_.to(device)
            
            # Initial states (t = 0) of all trajectories in the batch
            inp = input_batch[0::time_steps] 
            stats = model.loader_dictionary
            
            # Fields in physical units (for the energy diagnostic)
            c_phys = (inp[:, 1:2, :, :] * stats['std_c']) + stats['mean_c']
            u_prev_s_phys = (inp[:, 0:1, :, :] * stats['std_u']) + stats['mean_u']
            u_prev_gt_phys = u_prev_s_phys.clone()
            
            if physics_diagnostics:
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
                
                # One model step of size `jump`
                t_val = torch.zeros_like(t_batch[0::time_steps]) + jump * delta
                pred_norm = model(inp, t_val)
                out_norm = output_batch[acc_jump - 1::time_steps]

                u_s_phys = (pred_norm[:, 0:1] * stats['std_u']) + stats['mean_u']
                u_gt_phys = (out_norm[:, 0:1] * stats['std_u']) + stats['mean_u']

                if physics_diagnostics:
                    batch_energy_s.append(compute_energy(u_s_phys, u_prev_s_phys, c_phys, dx=DX_PHYS, dt_neural=DT_NEURAL))
                    batch_energy_gt.append(compute_energy(u_gt_phys, u_prev_gt_phys, c_phys, dx=DX_PHYS, dt_neural=DT_NEURAL))
                    shifts = [compute_phase_shift(pred_norm[:, 0:1][b], out_norm[:, 0:1][b]) for b in range(curr_b_size)]
                    batch_phases.append(torch.tensor(shifts))

                # Previous states for the next time derivative
                u_prev_s_phys, u_prev_gt_phys = u_s_phys.clone(), u_gt_phys.clone()

                # Relative L1 / L2 errors (in %) and absolute error
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

                # Feed the predicted wavefield back in (autoregressive rollout)
                inp[:, 0:1] = pred_norm[:, 0:1]

            if physics_diagnostics:
                energy_history_s.append(torch.stack(batch_energy_s).cpu().numpy())
                energy_history_gt.append(torch.stack(batch_energy_gt).cpu().numpy())
                phase_history.append(torch.stack(batch_phases).cpu().numpy())
            
            E_l1 = np.concatenate((E_l1, E_curr_l1), axis=-1)
            E_l2 = np.concatenate((E_l2, E_curr_l2), axis=-1)
            E_abs = np.concatenate((E_abs, E_curr_abs), axis=-1)

    if physics_diagnostics:
        full_e_s = np.concatenate(energy_history_s, axis=1)
        full_e_gt = np.concatenate(energy_history_gt, axis=1)
        plot_comparative_energy(full_e_s, full_e_gt, os.path.join(folder, "energy_decay_comparison.pdf"))
        plot_energy_consistency_ratio(full_e_s, full_e_gt, os.path.join(folder, "energy_consistency_ratio.pdf"))
        plot_phase_drift(np.concatenate(phase_history, axis=1), os.path.join(folder, "phase_drift.pdf"))
    
    return T, E_l1, E_l2, E_abs

def _evaluate_models(folders, which_example="wave_gauss", cluster=True, in_dist=True,
                     patterns=[[1]], folder_pattern="errors", steps=15, separate_var=False, 
                     fine_tuned=False, training_folder=None, write_in_file=True, physics_diagnostics=False):
    
    base_folder = str(paths.FORWARD_CHECKPOINT_DIR)
    ckpt_file = paths.FORWARD_CKPT
    model_dirs = [(base_folder, ckpt_file)]
    suffix = "in_dist_patterns.txt" if in_dist else "out_dist_patterns.txt"
    
    for folder, _model_file in model_dirs:
        _folder = folder if not fine_tuned and training_folder is None else training_folder
        m_dir = paths.FORWARD_RUN
        
        train_file = os.path.join(_folder, m_dir, "training_properties.txt")
        net_file   = os.path.join(_folder, m_dir, "net_architecture.txt")
        ckpt_path  = os.path.join(folder, m_dir, _model_file)
        
        loader = WaveLoader(train_file, net_file, ckpt_path, which_name=which_example)
        
        # Add the dataset normalization statistics to the configuration before building the model
        loader._ensure_config()
        stats = load_json_stats(str(paths.DATA_DIR))
        loader.loader_dict.update(stats)
        model = loader.load_model()

        # Evaluate on the full 120-trajectory test set
        loader.loader_dict["num_test"] = 120 
        loader.loader_dict["num_trajectories"] = 120
            
        bs = 32 if 32 % steps == 0 else (32 // steps) * steps
        if bs == 0: bs = steps
        
        test_data_loader = _load_data(which=which_example, loader_dict=loader.loader_dict, 
                                     batch_size=bs, which_loader="test", fix_input_to_time_step=0) 
        
        output_dir = os.path.join(paths.RESULTS_DIR, "forward", folder_pattern)
        os.makedirs(output_dir, exist_ok=True)
        
        for pattern in patterns:
            T, E_l1, E_l2, E_abs = _test_pattern(model=model, testing_loader=test_data_loader, 
                                                dt=loader.loader_dict["dt"], time_steps=loader.loader_dict["time_steps"], 
                                                max_steps=loader.loader_dict["step_max"], pattern=pattern, folder=output_dir,
                                                device=loader.device, batch_size=bs, separate_var=separate_var, 
                                                separate_dim=loader.loader_dict.get("separate_dim", []),
                                                physics_diagnostics=physics_diagnostics)
            
            p_str = "".join([str(p) + "_" for p in pattern])
            error_file = os.path.join(output_dir, p_str + suffix)
            if write_in_file:
                _write_in_file(error_file, loader.loader_dict, E_l1, E_l2, E_abs, T, pattern, 
                              separate_var=separate_var, num_var=len(loader.loader_dict.get("separate_dim", [])))
            print(f"Results for pattern {pattern} saved to {error_file}")

if __name__ == "__main__":
    base_checkpoints_dir = str(paths.FORWARD_CHECKPOINT_DIR)
    target_example = "wave_gauss"
    
    # 15 autoregressive steps of size 1, i.e. errors at every time step
    diagnostic_patterns = [[1] * 15]

    # Also track energy and phase drift (slower; phase drift is Fig. 2 of the report)
    PHYSICS_DIAGNOSTICS = False

    print("Forward CNO: autoregressive rollout on the 120 test trajectories")

    _evaluate_models(
        folders=base_checkpoints_dir,
        which_example=target_example,
        cluster=True,
        in_dist=True,
        patterns=diagnostic_patterns,
        folder_pattern="full_physics_diagnostic",
        steps=15, 
        separate_var=True,
        write_in_file=True,
        physics_diagnostics=PHYSICS_DIAGNOSTICS
    )
    print("Done.")
    print("="*60)