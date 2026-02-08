import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

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
        
        if "seismic" in self.loader.which_name:
            self.constants = { "mean": 0.03467443221585092, "std": 0.10442421752963911, "mean_c": 3498.5644380917424, "std_c": 647.843958567462, "time_max": 20.0, "step_max": 20, "filename": "wave_layer_data.nc" }
        else:
            self.constants = { "mean": 0.0334376316, "std": 0.1171879068, "mean_c": 2618.4593933, "std_c": 601.51658913, "time_max": 15.0, "step_max": 15, "filename": "wave_gauss_data.nc" }

        data_path = f"{PROJECT_ROOT}/data/{self.constants['filename']}"
        try: self.reader = h5py.File(data_path, "r")
        except FileNotFoundError: print(f"WARNING: Could not find data file at {data_path}")

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
        print(f"Generating autoregressive rollout for trajectory {traj_idx}...")
        
        u_curr_phys = self.reader["u"][traj_idx, 0] 
        if u_curr_phys.ndim == 3: u_curr_phys = u_curr_phys.squeeze()
        c_field_phys = self.reader["c"][traj_idx]   
        if c_field_phys.ndim == 3: c_field_phys = c_field_phys.squeeze()

        preds = [u_curr_phys]
        
        # --- STATIC FIELDS (Constant throughout rollout) ---
        _, c_norm_np = self._normalize(u_curr_phys, c_field_phys)
        c_tensor = torch.from_numpy(c_norm_np).float().to(self.device).unsqueeze(0) # Channel 1: Velocity
        
        # Channel 2: Time (Step Size)
        dt_normalized = 1.0 / self.constants["time_max"]
        t_scalar = torch.tensor([[dt_normalized]], dtype=torch.float32).to(self.device)
        
        # The Time MAP must also be constant dt
        t_channel = torch.ones_like(c_tensor) * dt_normalized 

        current_u_phys = u_curr_phys
        
        for t_step in range(1, self.constants["step_max"] + 1):
            
            # Channel 0: Dynamic Wave
            u_norm_np, _ = self._normalize(current_u_phys, c_field_phys)
            u_tensor = torch.from_numpy(u_norm_np).float().to(self.device).unsqueeze(0) 
            
            # Input: [u_dynamic, c_STATIC, dt_STATIC]
            x_in = torch.cat([u_tensor, c_tensor, t_channel], dim=0).unsqueeze(0) 
            
            with torch.no_grad():
                u_next_norm = self.model(x_in, t_scalar) 
            
            u_next_wave_norm_2d = u_next_norm[0, 0, :, :].cpu().numpy()
            u_next_phys = self._denormalize(u_next_wave_norm_2d)
            preds.append(u_next_phys)
            current_u_phys = u_next_phys

        return preds, c_field_phys

    def get_rollout_data(self, traj_idx):
        return self.autoregressive_rollout(traj_idx)

    # --- PLOTTING METHODS ---

    def plot_evolution_grid(self, traj_idx=0, save_path="evolution_grid.png", precomputed_data=None):
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
        print(f"Reading error statistics from: {error_file_path}")
        data = { "median": {}, "mean": {}, "median_l1": {}, "mean_l1": {} }
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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors, labels = ['b', 'r'], ['Wave (u)', 'Velocity (c)']
        for var_idx in [0, 1]:
            if var_idx in data['median']:
                times = sorted(data['median'][var_idx].keys())
                ax1.plot(times, [data['median'][var_idx][t] for t in times], label=f"{labels[var_idx]} Median", color=colors[var_idx], marker='o')
                ax1.plot(times, [data['mean'][var_idx][t] for t in times], label=f"{labels[var_idx]} Mean", color=colors[var_idx], linestyle='--', alpha=0.6)
        ax1.set_title("Relative L1 Error (%)"); ax1.set_xlabel("Time Step"); ax1.grid(True, alpha=0.3); ax1.legend()

        for var_idx in [0, 1]:
            if var_idx in data['median_l1']:
                times = sorted(data['median_l1'][var_idx].keys())
                ax2.plot(times, [data['median_l1'][var_idx][t] for t in times], label=f"{labels[var_idx]} Median", color=colors[var_idx], marker='o')
                ax2.plot(times, [data['mean_l1'][var_idx][t] for t in times], label=f"{labels[var_idx]} Mean", color=colors[var_idx], linestyle='--', alpha=0.6)
        ax2.set_title("Absolute L1 Error"); ax2.set_xlabel("Time Step"); ax2.grid(True, alpha=0.3); ax2.legend()
        plt.suptitle(f"Aggregated Performance: {os.path.basename(error_file_path)}", fontsize=14); plt.tight_layout()
        if save_path: plt.savefig(save_path); print(f"Saved aggregated plot to {save_path}")
        plt.close(fig)

    def plot_relative_l1_error(self, traj_idx=0, save_dir=f"{PROJECT_ROOT}/results", precomputed_data=None):
        if precomputed_data: preds, _ = precomputed_data
        else: preds, _ = self.autoregressive_rollout(traj_idx)
        ground_truth = self.reader["u"][traj_idx, 0:len(preds)]
        l1_errors = []
        for t in range(len(preds)):
            diff = np.mean(np.abs(preds[t] - ground_truth[t]))
            true_mean = np.mean(np.abs(ground_truth[t])) + 1e-10
            l1_errors.append((diff / true_mean) * 100)
        plt.figure(figsize=(10, 6)); plt.plot(range(len(preds)), l1_errors, marker='o', color='r')
        plt.title(f"Relative L1 Error (Trajectory {traj_idx})"); plt.xlabel("Time Step"); plt.ylabel("Error (%)"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"error_plot_{traj_idx}.png")); plt.close()

    def create_error_comparison_gif(self, traj_idx=0, save_dir=f"{PROJECT_ROOT}/results", fps=3, precomputed_data=None):
        if precomputed_data: preds, c_field = precomputed_data
        else: preds, c_field = self.autoregressive_rollout(traj_idx)
        ground_truth = self.reader["u"][traj_idx, 0:len(preds)]
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
    base_dir = f"{PROJECT_ROOT}/results/checkpoints/model_1401"
    loader = WaveLoader(os.path.join(base_dir, "training_properties.txt"), os.path.join(base_dir, "net_architecture.txt"), os.path.join(base_dir, "epoch=45-step=100096.ckpt"), which_name="wave_gauss")
    viz = WaveVisualizer(loader)
    
    err_file = f"{base_dir}/../errors/1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_in_dist_patterns.txt"
    if os.path.exists(err_file): viz.plot_dataset_errors(err_file, save_path=f"{PROJECT_ROOT}/results/aggregated_errors.png")

    rng = np.random.default_rng(seed=69)
    # for idx in rng.integers(10272, 10511, size=1):
    for idx in np.arange(0,10):
        # idx=0
        print(f"Visualizing trajectory {idx}...")
        data = viz.get_rollout_data(idx)
        viz.plot_relative_l1_error(idx, precomputed_data=data)
        viz.create_error_comparison_gif(idx, precomputed_data=data)
        viz.plot_evolution_grid(traj_idx=idx, save_path=f"{PROJECT_ROOT}/results/evolution_grid_{idx}.png", precomputed_data=data)
        viz.create_evolution_gif(traj_idx=idx, out_path=f"{PROJECT_ROOT}/results/evolution_{idx}.gif", precomputed_data=data)