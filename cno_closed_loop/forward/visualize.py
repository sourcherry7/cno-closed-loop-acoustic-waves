import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import json

# --- SETUP PATHS ---
PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
EXTERNAL_CNO_DIR = f"{PROJECT_ROOT}/src/external/CNO/CNO2d_time_dependent_&_foundation_model"
sys.path.append(EXTERNAL_CNO_DIR)
sys.path.append(f"{PROJECT_ROOT}/src/my_models")

class WaveVisualizer:
    def __init__(self, wave_loader):
        self.loader = wave_loader
        self.model = self.loader.load_model()
        self.device = self.loader.device
        
        # DYNAMIC LOADING: Read from your new JSON
        stats_path = os.path.join(PROJECT_ROOT, "data/new_normalization_stats.json")
        with open(stats_path, 'r') as f:
            js = json.load(f)
        
        self.constants = {
            "mean": js['mean_u'],
            "std": js['std_u'],
            "mean_c": js['mean_c'],
            "std_c": js['std_c'],
            "time_max": float(js['time_steps']),
            "step_max": js['time_steps'],
            "filename": "new_wave_gauss_data.h5" # Point to new H5
        }
        data_path = f"{PROJECT_ROOT}/data/{self.constants['filename']}"
        try: self.reader = h5py.File(data_path, "r")
        except FileNotFoundError: print(f"WARNING: Could not find data file at {data_path}")

        total_trajectories = self.reader['u'].shape[0]
        self.num_test = 120
        
        # Calculate the starting point of the test set
        self.test_start_idx = total_trajectories - self.num_test
        print(f"[INFO] Visualizer targeting test set starting at index {self.test_start_idx}")

    def _normalize(self, u, c):
        u_norm = (u - self.constants["mean"]) / self.constants["std"]
        c_norm = (c - self.constants["mean_c"]) / self.constants["std_c"]
        return u_norm, c_norm

    def _denormalize(self, u_norm, c_norm=None):
        u = u_norm * self.constants["std"] + self.constants["mean"]
        if c_norm is not None:
            c = c_norm * self.constants["std_c"] + self.constants["mean_c"]
            return u, c
        return u

    def autoregressive_rollout(self, traj_idx=0):
        """
        The 'Engine': Generates a 15-step rollout from the test set.
        """
        global_idx = self.test_start_idx + traj_idx
        print(f"Generating rollout for Test Trajectory {traj_idx} (Global H5 Index {global_idx})...")
        
        # 1. Load initial physical state
        u_curr_phys = self.reader["u"][global_idx, 0] 
        if u_curr_phys.ndim == 3: u_curr_phys = u_curr_phys.squeeze()
        c_field_phys = self.reader["c"][global_idx]   
        if c_field_phys.ndim == 3: c_field_phys = c_field_phys.squeeze()

        preds = [u_curr_phys]
        
        # 2. Prepare STATIC components (Velocity and Time Map)
        _, c_norm_np = self._normalize(u_curr_phys, c_field_phys)
        
        # Shape: [1, 1, 128, 128]
        c_tensor = torch.from_numpy(c_norm_np).float().to(self.device).unsqueeze(0).unsqueeze(0)
        
        dt_normalized = 1.0 / self.constants["time_max"]
        # t_scalar is the scalar input for the Time-dependent CNO [1, 1]
        t_scalar = torch.tensor([[dt_normalized]], dtype=torch.float32).to(self.device)
        # t_channel is the spatial map input (Channel 2) [1, 1, 128, 128]
        t_channel = torch.ones_like(c_tensor) * dt_normalized 

        current_u_phys = u_curr_phys
        
        # 3. Autoregressive Loop
        for t_step in range(1, self.constants["step_max"] + 1):
            # Normalize and prepare the dynamic wave channel [1, 1, 128, 128]
            u_norm_np, _ = self._normalize(current_u_phys, c_field_phys)
            u_tensor = torch.from_numpy(u_norm_np).float().to(self.device).unsqueeze(0).unsqueeze(0) 
            
            # CONCATENATE ALONG DIM 1 -> Result: [1, 3, 128, 128]
            # This aligns exactly with the model's 'in_dim=3' requirement.
            x_in = torch.cat([u_tensor, c_tensor, t_channel], dim=1)
            
            with torch.no_grad():
                # Forward Pass: model(Input_Tensor, Time_Scalar)
                u_next_norm = self.model(x_in, t_scalar) 
            
            # Extract and denormalize wavefield (Channel 0)
            u_next_wave_norm_2d = u_next_norm[0, 0, :, :].cpu().numpy()
            u_next_phys = self._denormalize(u_next_wave_norm_2d)
            
            preds.append(u_next_phys)
            current_u_phys = u_next_phys

        return preds, c_field_phys

    def get_rollout_data(self, traj_idx):
        return self.autoregressive_rollout(traj_idx)

    # --- PLOTTING METHODS ---

    def plot_evolution_grid(self, traj_idx=0, save_path="evolution_grid.pdf", precomputed_data=None):
        if precomputed_data: preds, c_field = precomputed_data
        else: preds, c_field = self.autoregressive_rollout(traj_idx)
        
        snapshots = preds[0:16] # Use t=0 to t=15
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(5, 4) 
        v_min, v_max = np.min(preds), np.max(preds)

        for i, frame in enumerate(snapshots):
            if i >= 16: break
            row, col = i // 4, i % 4
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(frame, cmap='RdBu_r', vmin=v_min, vmax=v_max)
            ax.set_title(f"t = {i}"); ax.axis('off')
            if col == 3: plt.colorbar(im, ax=ax, fraction=0.046)
            
        ax_c = fig.add_subplot(gs[4, 1:3])
        im_c = ax_c.imshow(c_field, cmap='viridis')
        ax_c.set_title("Velocity Field (c)"); ax_c.axis('off'); plt.colorbar(im_c, ax=ax_c, fraction=0.046)
        plt.suptitle(f"Wave Evolution (Sample {traj_idx})", fontsize=20); plt.tight_layout()
        plt.savefig(save_path); plt.close(fig)
        print(f"Saved grid to {save_path}")

    def create_evolution_gif(self, traj_idx=0, out_path="evolution.gif", fps=5, precomputed_data=None):
        if precomputed_data: preds, c_field = precomputed_data
        else: preds, c_field = self.autoregressive_rollout(traj_idx)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im_c = ax2.imshow(c_field, cmap='viridis'); ax2.set_title("Velocity (c)"); ax2.axis('off'); plt.colorbar(im_c, ax=ax2, fraction=0.046)
        
        v_min, v_max = np.min(preds), np.max(preds)
        im_wave = ax1.imshow(preds[0], cmap='RdBu_r', vmin=v_min, vmax=v_max)
        title = ax1.set_title("t = 0"); ax1.axis('off'); plt.colorbar(im_wave, ax=ax1, fraction=0.046)

        def update(frame_idx):
            im_wave.set_data(preds[frame_idx]); title.set_text(f"Wave (t = {frame_idx})")
            return [im_wave, title]

        ani = animation.FuncAnimation(fig, update, frames=len(preds), interval=200, blit=False)
        ani.save(out_path, writer=animation.PillowWriter(fps=fps)); plt.close(fig)
        print(f"Saved evolution GIF to {out_path}")

    def plot_dataset_errors(self, error_file_path, save_path=None):
        """
        Consolidates Rel L1 and L2 with standard deviation shading into a single report figure.
        Prints a summary table for the final rollout step.
        """
        print(f"Reading comprehensive error statistics from: {error_file_path}")
        # Initialize dictionary to handle new std metrics
        data = { 
            "median_l1": {}, "mean_l1": {}, "std_l1": {},
            "median_l2": {}, "mean_l2": {}, "std_l2": {},
            "median_abs": {}, "mean_abs": {} 
        }
        
        with open(error_file_path, 'r') as f:
            for line in f:
                try:
                    left, value = line.strip().split(',')
                    value = float(value)
                    parts = left.split(' ')
                    metric, time_var = parts[0], parts[1] 
                    
                    t_str, var_str = time_var.split('_')
                    t, var = int(t_str), int(var_str)
                    
                    if var not in data[metric]: data[metric][var] = {}
                    data[metric][var][t] = value
                except Exception: continue

        # --- 1. PLOTTING ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Style configuration for report clarity
        styles = {
            (0, 'l1'): {'color': 'blue', 'ls': '-',  'label': 'Wave $u$ (Rel. $L^1$ %)'},
            (0, 'l2'): {'color': 'blue', 'ls': '--', 'label': 'Wave $u$ (Rel. $L^2$ %)'},
            (1, 'l1'): {'color': 'red',  'ls': '-',  'label': 'Velocity $c$ (Rel. $L^1$ %)'},
            (1, 'l2'): {'color': 'red',  'ls': '--', 'label': 'Velocity $c$ (Rel. $L^2$ %)'}
        }

        for var_idx in [0, 1]: # 0=Wave, 1=Velocity
            for m_key in ['l1', 'l2']:
                mean_key = f"mean_{m_key}"
                std_key = f"std_{m_key}"
                
                if var_idx in data[mean_key] and var_idx in data[std_key]:
                    times = sorted(data[mean_key][var_idx].keys())
                    means = np.array([data[mean_key][var_idx][t] for t in times])
                    stds  = np.array([data[std_key][var_idx][t] for t in times])
                    
                    style = styles[(var_idx, m_key)]
                    
                    # Plot the Central Line (Mean)
                    ax.plot(times, means, linewidth=2, **style)
                    
                    # Plot the Shaded Region (Mean +/- 1 Std Dev)
                    ax.fill_between(times, means - stds, means + stds, 
                                    color=style['color'], alpha=0.15, lw=0)

        # Remove axes titles and graph title as requested
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
        ax.legend(frameon=True, loc='upper left', fontsize=11)
        
        # Use tight layout with minimal padding
        plt.tight_layout(pad=0.2)
        
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved aggregated report plot with shading to {save_path}")
        plt.close(fig)

        # --- 2. SUMMARY TABLE ---
        self._print_error_table(data)

    def plot_dataset_l2_errors(self, error_file_path, save_path=None):
        """
        Plots ONLY the Relative L2 errors with standard deviation shading.
        Optimized for scientific reports: no big title, minimal padding, axis legends
        """
        print(f"Reading L2 error statistics from: {error_file_path}")
        # Initialize dictionary to handle new std metrics
        data = { 
            "median_l1": {}, "mean_l1": {}, "std_l1": {},
            "median_l2": {}, "mean_l2": {}, "std_l2": {},
            "median_abs": {}, "mean_abs": {} 
        }
        
        with open(error_file_path, 'r') as f:
            for line in f:
                try:
                    left, value = line.strip().split(',')
                    value = float(value)
                    parts = left.split(' ')
                    metric, time_var = parts[0], parts[1] 
                    
                    t_str, var_str = time_var.split('_')
                    t, var = int(t_str), int(var_str)
                    
                    if var not in data[metric]: data[metric][var] = {}
                    data[metric][var][t] = value
                except Exception: continue

        # --- 1. PLOTTING ---
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Style configuration for L2-only report plot
        # Color: Blue for Wave (u), Red for Velocity (c)
        styles = {
            0: {'color': 'blue', 'ls': '-', 'marker': 's', 'label': 'Wave $u$'},
            1: {'color': 'red',  'ls': '-', 'marker': 's', 'label': 'Velocity $c$'}
        }

        # Focus strictly on L2
        m_key = 'l2'
        for var_idx in [0, 1]: # 0=Wave, 1=Velocity
            mean_key = f"mean_{m_key}"
            std_key = f"std_{m_key}"
            
            if var_idx in data[mean_key] and var_idx in data[std_key]:
                times = sorted(data[mean_key][var_idx].keys())
                means = np.array([data[mean_key][var_idx][t] for t in times])
                stds  = np.array([data[std_key][var_idx][t] for t in times])
                
                style = styles[var_idx]
                
                # Plot the Central Line (Mean)
                ax.plot(times, means, linewidth=2, **style)
                
                # Plot the Shaded Region (Mean +/- 1 Std Dev)
                ax.fill_between(times, means - stds, means + stds, 
                                color=style['color'], alpha=0.15, lw=0)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Relative $L2$ error (%)")
        # Report aesthetics
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
        ax.legend(frameon=True, loc='upper left', fontsize=11)
        
        # Tight layout with minimal padding and no text labels on axes/titles
        plt.tight_layout(pad=0.1)
        
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved L2-only report plot to {save_path}")
        plt.close(fig)

        # --- 2. SUMMARY TABLE ---
        # Note: We still print the full table so you can see L1/L2/Abs numbers in console
        self._print_error_table(data)

    def _print_error_table(self, data):
        """
        Prints a formatted table of mean errors and standard deviations 
        for all variables at the final rollout step.
        """
        # 1. Find the final time step available in the parsed data
        all_times = []
        for metric in data:
            for var in data[metric]:
                all_times.extend(data[metric][var].keys())
        
        if not all_times: 
            print("[Table] No error data found to print.")
            return
            
        t_final = max(all_times)
        
        # 2. Header Configuration (Total width ~95 characters)
        print("\n" + "="*95)
        print(f"{'VARIABLE':<15} | {'L1 MEAN (%)':<12} | {'L1 STD':<10} | {'L2 MEAN (%)':<12} | {'L2 STD':<10} | {'MEAN ABS':<10}")
        print("-" * 95)
        
        # 3. Data Rows
        labels = {0: "Wave (u)", 1: "Velocity (c)"}
        for var_idx in sorted(labels.keys()):
            try:
                # Extract Means
                l1_mean = data['mean_l1'][var_idx][t_final]
                l2_mean = data['mean_l2'][var_idx][t_final]
                abs_mean = data['mean_abs'][var_idx][t_final]
                
                # Extract Standard Deviations (safely handle missing keys for old files)
                l1_std = data.get('std_l1', {}).get(var_idx, {}).get(t_final, 0.0)
                l2_std = data.get('std_l2', {}).get(var_idx, {}).get(t_final, 0.0)
                
                # Print row with alignment
                print(f"{labels[var_idx]:<15} | {l1_mean:<12.4f} | {l1_std:<10.4f} | {l2_mean:<12.4f} | {l2_std:<10.4f} | {abs_mean:<10.6f}")
                
            except KeyError:
                # Skip if a variable (like Velocity c) wasn't tracked in that specific file
                continue
                
        print("="*95 + "\n")

    def plot_rollout_errors(self, traj_idx=0, save_dir=None, precomputed_data=None):
        """Calculates and plots both L1 and L2 for a single rollout."""
        print("\n Plotting both L1 and L2 for a single rollout...")
        if precomputed_data: preds, _ = precomputed_data
        else: preds, _ = self.autoregressive_rollout(traj_idx)
        
        # --- NEW: MAP TO TEST SET ---
        global_idx = self.test_start_idx + traj_idx
        ground_truth = self.reader["u"][global_idx, 0:len(preds)]
        l1_rel, l2_rel = [], []
        eps = 1e-10

        for t in range(len(preds)):
            diff = preds[t] - ground_truth[t]
            # Relative L1
            l1 = (np.mean(np.abs(diff)) / (np.mean(np.abs(ground_truth[t])) + eps)) * 100
            # Relative L2
            l2 = (np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(ground_truth[t]**2)) + eps)) * 100
            l1_rel.append(l1)
            l2_rel.append(l2)

        plt.figure(figsize=(10, 6))
        plt.plot(l1_rel, marker='o', label='Rel. L1 Error', color='blue')
        plt.plot(l2_rel, marker='s', label='Rel. L2 Error', color='red')
        plt.title(f"Accumulated Rollout Error (Trajectory {traj_idx})")
        plt.xlabel("Autoregressive Step"); plt.ylabel("Error (%)")
        plt.legend(); plt.grid(True, alpha=0.3)
        
        fname = f"rollout_error_{traj_idx}.pdf"
        save_path = os.path.join(save_dir, fname) if save_dir else fname
        plt.savefig(save_path); plt.close()
        print(f"Saved {fname} to {save_path}.")

    def create_error_comparison_gif(self, traj_idx=0, save_dir=f"{PROJECT_ROOT}/fwd_results", fps=3, precomputed_data=None):
        if precomputed_data: preds, c_field = precomputed_data
        else: preds, c_field = self.autoregressive_rollout(traj_idx)

        global_idx = self.test_start_idx + traj_idx
        ground_truth = self.reader["u"][global_idx, 0:len(preds)]

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        ax_pred, ax_true, ax_err, ax_c = axes
        v_min, v_max = np.min(ground_truth), np.max(ground_truth)
        err_max = np.max(np.abs(np.array(preds) - ground_truth))
        im_p = ax_pred.imshow(preds[0], cmap='jet', vmin=v_min, vmax=v_max); ax_pred.set_title("Prediction u_pred"); plt.colorbar(im_p, ax=ax_pred, fraction=0.046)
        im_t = ax_true.imshow(ground_truth[0], cmap='jet', vmin=v_min, vmax=v_max); ax_true.set_title("Ground Truth u_sol"); plt.colorbar(im_t, ax=ax_true, fraction=0.046)
        im_e = ax_err.imshow(np.abs(preds[0]-ground_truth[0]), cmap='magma', vmin=0, vmax=err_max); ax_err.set_title("Absolute Error |u_pred - u_sol|"); plt.colorbar(im_e, ax=ax_err, fraction=0.046)
        c_min, c_max = np.min(c_field), np.max(c_field)
        im_c = ax_c.imshow(c_field, cmap='jet', vmin=c_min, vmax=c_max); ax_c.set_title("Velocity Field (c)"); plt.colorbar(im_c, ax=ax_c, fraction=0.046)
        for ax in [ax_true, ax_err, ax_c]: ax.axis('off')
        ax_pred.tick_params(left=False, bottom=False, labelleft=False)
        N = len(preds)
        def update(t):
            im_p.set_data(preds[t]); im_t.set_data(ground_truth[t]); im_e.set_data(np.abs(preds[t] - ground_truth[t])); ax_pred.set_xlabel(f"t={t}")
            ax_pred.set_xlabel(f"T = {t}/{N-1}") 
            return [im_p, im_t, im_e, im_c]
        ani = animation.FuncAnimation(fig, update, frames=len(preds), interval=200, blit=True)
        ani.save(os.path.join(save_dir, f"error_comparison_{traj_idx}.gif"), writer=animation.PillowWriter(fps=fps)); plt.close(fig)
        print(f"Saved comparison GIF to {save_dir}")

if __name__ == "__main__":
    from wave_loaders import WaveLoader
    base_dir = f"{PROJECT_ROOT}/fwd_results/checkpoints/model_0903_newgauss"
    plot_dir = base_dir + "/../visuals_0903_newgauss"
    loader = WaveLoader(os.path.join(base_dir, "training_properties.txt"), os.path.join(base_dir, "net_architecture.txt"), os.path.join(base_dir, "epoch=20-step=182784.ckpt"), which_name="wave_gauss")
    viz = WaveVisualizer(loader)
    
    err_file = f"{base_dir}/../full_physics_diagnostic/1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_in_dist_patterns.txt"
    if os.path.exists(err_file): 
        viz.plot_dataset_errors(err_file, save_path=f"{plot_dir}/aggregated_errors.pdf")
        viz.plot_dataset_l2_errors(err_file, save_path=f"{plot_dir}/aggregated_l2_errors.pdf")


    for idx in np.arange(0,2):
        # idx=0
        print(f"Visualizing trajectory {idx}...")
        data = viz.get_rollout_data(idx)
        viz.plot_rollout_errors(idx, precomputed_data=data, save_dir=plot_dir)
        viz.create_error_comparison_gif(idx, precomputed_data=data, save_dir=plot_dir)
        viz.plot_evolution_grid(traj_idx=idx, save_path=f"{plot_dir}/evolution_grid_{idx}.pdf", precomputed_data=data)
        viz.create_evolution_gif(traj_idx=idx, out_path=f"{plot_dir}/evolution_{idx}.gif", precomputed_data=data)