import os
import sys
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
MYMODELS_DIR = f"{PROJECT_ROOT}/src/my_models"
INV_DIR = f"{MYMODELS_DIR}/inverse_problem"
sys.path.append(INV_DIR)

from inv_CNO_timeModule_CIN import CNO_time
from inv_wave_loader import InverseWaveGaussians
from inv_utils import _load_inverse_dict

# -----------------------------------------------------------------------------
# SMART FIX: GLOBAL CUDA BYPASS
# -----------------------------------------------------------------------------
# Intercepts the hardcoded CNO functional call in memory to prevent 
# JIT C++ compilation and force the native PyTorch reference implementation.
def patch_cno_cuda_ops():
    for module_name, module in sys.modules.items():
        if 'ops.filtered_lrelu' in module_name and hasattr(module, 'filtered_lrelu'):
            original_fn = module.filtered_lrelu
            def safe_filtered_lrelu(*args, **kwargs):
                kwargs['impl'] = 'ref'
                return original_fn(*args, **kwargs)
            module.filtered_lrelu = safe_filtered_lrelu

patch_cno_cuda_ops()

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def safe_eval(val):
    if isinstance(val, str):
        val = val.strip()
        try: return ast.literal_eval(val)
        except: return val
    return val

# -----------------------------------------------------------------------------
# PLOTTING & METRICS
# -----------------------------------------------------------------------------
def plot_test_sample(model, dataset, sample_idx, save_path):
    model.eval()
    device = next(model.parameters()).device
    
    time_val, input_tensor, target_c = dataset[sample_idx]
    time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch, time_tensor)
    
    target_np = target_c[0].cpu().numpy()
    pred_np = prediction[0, 0].cpu().numpy()
    
    c_min, c_max = np.min(target_np), np.max(target_np)
    u_min = min(np.min(input_tensor[0].numpy()), np.min(input_tensor[1].numpy()))
    u_max = max(np.max(input_tensor[0].numpy()), np.max(input_tensor[1].numpy()))
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    dt_actual = time_val * 15.0
    
    im0 = ax[0].imshow(input_tensor[0].numpy(), vmin=u_min, vmax=u_max, cmap='jet')
    ax[0].set_title("Input: u(t1)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    
    im1 = ax[1].imshow(input_tensor[1].numpy(), vmin=u_min, vmax=u_max, cmap='jet')
    ax[1].set_title(f"Input: u(t2) (dt={dt_actual:.1f})")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
    # im2 = ax[2].imshow(input_tensor[1].numpy() - input_tensor[0].numpy(), cmap='RdBu')
    # ax[2].set_title("Motion (u2 - u1)")
    # fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    
    im3 = ax[2].imshow(target_np, vmin=c_min, vmax=c_max, cmap='jet')
    ax[2].set_title("Ground truth velocity field")
    fig.colorbar(im3, ax=ax[2], fraction=0.046, pad=0.04)
    
    im4 = ax[3].imshow(pred_np, vmin=c_min, vmax=c_max, cmap='jet')
    ax[3].set_title("Predicted velocity field")
    fig.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_test_metrics(model, dataset, device):
    model.eval()
    metrics_by_dt = {}
    eps = 1e-8
    nan_count = 0
    
    print(f"Processing {len(dataset)} samples on {device}...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating Metrics"):
            time_val, input_tensor, target_c = dataset[idx]
            dt = round(time_val * 15.0) 
            
            time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
            input_batch = input_tensor.unsqueeze(0).to(device)
            target_c = target_c.to(device)
            
            prediction = model(input_batch, time_tensor)
            
            if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                nan_count += 1
                continue

            rel_l2 = (torch.norm(prediction - target_c, p=2) / (torch.norm(target_c, p=2) + eps)).item()
            rel_l1 = (torch.norm(prediction - target_c, p=1) / (torch.norm(target_c, p=1) + eps)).item()
            
            if dt not in metrics_by_dt: metrics_by_dt[dt] = {"l1": [], "l2": []}
            metrics_by_dt[dt]["l1"].append(rel_l1)
            metrics_by_dt[dt]["l2"].append(rel_l2)
            
    if nan_count > 0:
        print(f"WARNING: Skipped {nan_count} samples due to NaNs.")
    return metrics_by_dt

def summarize_and_save(metrics_dict, output_dir):
    if not metrics_dict:
        print("No valid metrics to summarize.")
        return

    summary = []
    all_l1, all_l2 = [], []
    dts = sorted(metrics_dict.keys())
    
    print(f"\n{'dt':<5} | {'Mean Rel. L1':<15} | {'Mean Rel. L2':<15}")
    print("-" * 45)
    for dt in dts:
        m_l1, m_l2 = np.mean(metrics_dict[dt]["l1"]), np.mean(metrics_dict[dt]["l2"])
        all_l1.extend(metrics_dict[dt]["l1"]); all_l2.extend(metrics_dict[dt]["l2"])
        print(f"{dt:<5.1f} | {m_l1:<15.4f} | {m_l2:<15.4f}")
        summary.append({"dt": dt, "rel_l1": m_l1, "rel_l2": m_l2})
    
    ov_l1, ov_l2 = np.mean(all_l1), np.mean(all_l2)
    print("-" * 45)
    print(f"{'TOTAL':<5} | {ov_l1:<15.4f} | {ov_l2:<15.4f}")
    
    summary.append({"dt": "OVERALL", "rel_l1": ov_l1, "rel_l2": ov_l2})
    pd.DataFrame(summary).to_csv(f"{output_dir}/test_metrics_summary.csv", index=False)
    
    plt.figure(figsize=(5, 4))
    plt.bar(dts, [np.mean(metrics_dict[d]["l2"]) for d in dts], color='#f28e2b')
    plt.xlabel("Time Step Gap (dt)"); plt.ylabel("Mean Relative L2 Error")
    plt.title("Error vs Snapshot Separation")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{output_dir}/error_by_dt.png")
    plt.close()


def make_test_dataset_from_ckpt(model_name, checkpoint_name, num_traj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    folder = "/cluster/home/svisnjic/Thalassa/inv_results/checkpoints/" + str(model_name) # "model_1802_second"
    ckpt_path = f"{folder}/{checkpoint_name}" # epoch=2-step=46080.ckpt
    data_path = "/cluster/home/svisnjic/Thalassa/data"
    
    # 1. Load Architecture
    files = [f"{folder}/inv_training_properties.txt", f"{folder}/inv_net_architecture.txt"]
    train_prop = pd.read_csv(files[0], header=None, index_col=0).squeeze().to_dict()
    net_arch = pd.read_csv(files[1], header=None, index_col=0).squeeze().to_dict()
    
    for d in [train_prop, net_arch]:
        for k, v in d.items(): d[k] = safe_eval(v)

    net_arch["batch_norm"] = net_arch.get("batch_norm") in [1, True, 'True']
    net_arch["is_att"] = net_arch.get("is_att", False) in [1, True, 'True']

    # 2. Setup Loader Dictionary
    loader_dict = _load_inverse_dict(files=files, steps=15, which_example='wave_gauss', is_masked=False)
    loader_dict['data_path'] = data_path
    
    _allowed = list(range(0 if loader_dict.get("include_zero") else 1, int(train_prop.get("time_steps", 15)) + 1))
    loader_dict['allowed_transitions'] = _allowed

    # 3. Model Initialization (RESTORED MISSING CONDITIONAL PARAMS)
    print(f"Initializing model on {device}...")
    model = CNO_time.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        in_dim=loader_dict["in_dim"],
        out_dim=loader_dict.get("out_dim", 1),
        in_size=net_arch.get("in_size", 128),
        N_layers=net_arch.get("N_layers"),
        channel_multiplier=net_arch.get("channel_multiplier"),
        N_res=net_arch.get("N_res"),
        N_res_neck=net_arch.get("N_res_neck"),
        batch_norm=net_arch.get("batch_norm"),
        is_time=net_arch.get("is_time", 1),
        is_att=net_arch.get("is_att"),
        activation=net_arch.get("activation", "cno_lrelu_torch"),
        nl_dim=net_arch.get("nl_dim"),
        time_steps=loader_dict["time_steps"],
        time_input=loader_dict.get('time_input', True),
        loader_dictionary=loader_dict,
        strict=False, 
        allowed_transitions=loader_dict['allowed_transitions']
    ).to(device)

    # 4. Dataset Loading
    test_dataset = InverseWaveGaussians(
        which="test", resolution=128, num_trajectories=num_traj, 
        data_path=data_path, max_num_time_steps=15, 
        time_step_size=1, allowed_transitions=loader_dict['allowed_transitions']
    )

    return model, test_dataset

# -----------------------------------------------------------------------------
# DIAGNOSTIC ADDITIONS
# -----------------------------------------------------------------------------

def plot_absolute_error_diagnostic(model, dataset, sample_idx, save_path):
    """Plots Ground Truth, Prediction, and Absolute Error Heatmap on a unified scale."""
    model.eval()
    device = next(model.parameters()).device
    
    time_val, input_tensor, target_c = dataset[sample_idx]
    time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch, time_tensor)
    
    target_np = target_c[0].cpu().numpy()
    pred_np = prediction[0, 0].cpu().numpy()
    abs_error = np.abs(target_np - pred_np)
    
    # Calculate global min/max for a unified color scale
    cmin = min(target_np.min(), pred_np.min())
    cmax = max(target_np.max(), pred_np.max())
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Use vmin/vmax to lock the scale for Truth and Prediction
    im0 = ax[0].imshow(target_np, cmap='jet', vmin=cmin, vmax=cmax)
    ax[0].set_title("Ground Truth $c$")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    
    im1 = ax[1].imshow(pred_np, cmap='jet', vmin=cmin, vmax=cmax)
    ax[1].set_title("Predicted $\hat{c}$")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
    # Absolute Error keeps its own scale to highlight local discrepancies
    im2 = ax[2].imshow(abs_error, cmap='inferno')
    ax[2].set_title("Absolute Error $|c - \hat{c}|$")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Diagnostic error plot saved to: {save_path}")

def plot_spectral_analysis(model, dataset, device, output_dir, num_samples=120):
    """Performs Radial Averaged Power Spectrum analysis on test samples."""
    model.eval()
    all_raps_true = []
    all_raps_pred = []
    
    print(f"Running spectral analysis on {num_samples} samples...")
    
    with torch.no_grad():
        # We take one sample per unique velocity field (dt=9 is usually optimal)
        for i in tqdm(range(num_samples)):
            idx = i * 120 + 8 # Targeting dt=9
            time_val, input_tensor, target_c = dataset[idx]
            
            time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
            input_batch = input_tensor.unsqueeze(0).to(device)
            prediction = model(input_batch, time_tensor)
            
            true_img = target_c[0].cpu().numpy()
            pred_img = prediction[0, 0].cpu().numpy()
            
            def get_raps(img):
                # 2D FFT and Power Spectrum
                f = np.fft.fft2(img)
                fshift = np.fft.fftshift(f)
                psd2d = np.abs(fshift)**2
                
                # Create radial grid
                h, w = img.shape
                yc, xc = h // 2, w // 2
                y, x = np.indices(psd2d.shape)
                r = np.sqrt((x - xc)**2 + (y - yc)**2)
                r = r.astype(int)
                
                # Average bins
                tbin = np.bincount(r.ravel(), psd2d.ravel())
                nr = np.bincount(r.ravel())
                radialprofile = tbin / nr
                return radialprofile[:h//2]

            all_raps_true.append(get_raps(true_img))
            all_raps_pred.append(get_raps(pred_img))

    # Convert to arrays and calculate statistics
    raps_true = np.array(all_raps_true)
    raps_pred = np.array(all_raps_pred)
    
    mean_true, std_true = np.mean(raps_true, axis=0), np.std(raps_true, axis=0)
    mean_pred, std_pred = np.mean(raps_pred, axis=0), np.std(raps_pred, axis=0)
    
    freqs = np.arange(len(mean_true))
    
    plt.figure(figsize=(6, 4))
    plt.plot(freqs, mean_true, label="Ground Truth", color="black", lw=2)
    plt.fill_between(freqs, mean_true - std_true, mean_true + std_true, color="black", alpha=0.1)
    
    plt.plot(freqs, mean_pred, label="CNO Prediction", color="#f28e2b", lw=2)
    plt.fill_between(freqs, mean_pred - std_pred, mean_pred + std_pred, color="#f28e2b", alpha=0.2)
    
    plt.yscale('log')
    plt.xlabel("Spatial Frequency (radial bin)")
    plt.ylabel("Power Spectral Density (Log Scale)")
    plt.title("Radial Averaged Power Spectrum: $c$ vs $\hat{c}$")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_path = f"{output_dir}/spectral_analysis_raps.pdf"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Spectral analysis plot saved to: {save_path}")

# -----------------------------------------------------------------------------
# UPDATED MAIN
# -----------------------------------------------------------------------------
def main(model_name, checkpoint_name, num_traj):
    model, test_dataset = make_test_dataset_from_ckpt(model_name=model_name, checkpoint_name=checkpoint_name, num_traj=num_traj)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    output_dir = "/cluster/home/svisnjic/Thalassa/inv_results/testing_visuals" + "/" + model_name
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Running Diagnostics ---")
    
    # 1. New Absolute Error Heatmap for a random sample (e.g., sample 500)
    
    # 2. New Spectral Analysis across all test trajectories
    plot_spectral_analysis(model, test_dataset, device, output_dir, num_samples=num_traj)
    
    # # Existing code for summary metrics and trajectory plots
    print("\nGenerating representative plots...")
    for traj in range(5): # Reduced for brevity
        for dt in [1, 9]:
            plot_absolute_error_diagnostic(model, test_dataset, traj * 120 + dt - 1, f"{output_dir}/diagnostic_abs_error_{traj}_dt_{dt}.pdf")
            # plot_test_sample(model, test_dataset, traj * 120 + dt - 1, f"{output_dir}/test_trajectory_{traj}_dt_{dt}.pdf")

    # metrics = calculate_test_metrics(model, test_dataset, device)
    # summarize_and_save(metrics, output_dir)
# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
# def main(model_name, checkpoint_name, num_traj):
#     model, test_dataset = make_test_dataset_from_ckpt(model_name=model_name, checkpoint_name = checkpoint_name, num_traj=num_traj)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device).eval()
#     # 5. Run Testing
#     output_dir = "/cluster/home/svisnjic/Thalassa/inv_results/testing_visuals" + "/" + model_name
#     os.makedirs(output_dir, exist_ok=True)
    
#     print("\nGenerating representative plots...")
    
#     for traj in range(5):
#         for dt in [1, 9]:
#             plot_test_sample(model, test_dataset, traj * 120 + dt - 1, f"{output_dir}/test_trajectory_{traj}_dt_{dt}.pdf")
#             print(f"Plotted trajctory {traj} at time pair (0, {dt}) and saved fig to: {output_dir}/test_trajectory_{traj}_dt_{dt}.pdf")

#     metrics = calculate_test_metrics(model, test_dataset, device)
#     summarize_and_save(metrics, output_dir)

    
#     df = pd.read_csv(f'{output_dir}/test_metrics_summary.csv')
#     df = df.iloc[:-1]
#     plt.figure(figsize=(6, 3.5))
#     plt.bar(df["dt"], df["rel_l2"], color='#f28e2b')
#     plt.xlabel("Time Step Gap (dt)")
#     plt.ylabel("Mean Relative L2 Error")
#     plt.title("Error vs Snapshot Separation")
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout(pad=1.1)
#     plt.savefig(f"{output_dir}/error_by_dt.pdf")
#     plt.close()
#     print(f'Saved error graph to {output_dir}/error_by_dt.pdf')


if __name__ == "__main__":
    main(model_name = "model_0103_new_data", checkpoint_name = "epoch=3-step=30720.ckpt", num_traj=120)