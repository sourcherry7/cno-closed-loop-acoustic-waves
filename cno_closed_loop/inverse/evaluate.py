"""Evaluation of the inverse CNO on the 120 test trajectories.

Produces the inverse-problem results of the report: relative L1/L2 errors per
snapshot gap dt (CSV + bar chart), example reconstructions, absolute-error maps
and the radially averaged power spectrum of true vs predicted velocity fields.
Run from the repository root:  python -m cno_closed_loop.inverse.evaluate
"""

import os
import sys
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from cno_closed_loop import paths
from cno_closed_loop.inverse.model import InverseCNO
from cno_closed_loop.inverse.dataset import InverseWaveGaussians
from cno_closed_loop.inverse.config import _load_inverse_dict

def patch_cno_cuda_ops():
    """Use the pure-PyTorch reference implementation of the CNO's filtered_lrelu,
    so that the custom CUDA kernel is not JIT-compiled at inference time."""
    for module_name, module in sys.modules.items():
        if 'ops.filtered_lrelu' in module_name and hasattr(module, 'filtered_lrelu'):
            original_fn = module.filtered_lrelu
            def safe_filtered_lrelu(*args, **kwargs):
                kwargs['impl'] = 'ref'
                return original_fn(*args, **kwargs)
            module.filtered_lrelu = safe_filtered_lrelu

patch_cno_cuda_ops()

def safe_eval(val):
    """Parse a value from a run property file (number, bool, list); keep plain strings."""
    if isinstance(val, str):
        val = val.strip()
        try: return ast.literal_eval(val)
        except (ValueError, SyntaxError): return val
    return val


def _radial_power_spectrum(img):
    """Radially averaged power spectral density of a 2D field (bins 0 .. H/2 - 1)."""
    # 2D FFT and power spectrum
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    psd2d = np.abs(fshift)**2

    # Radial distance of each frequency from the centre
    h, w = img.shape
    yc, xc = h // 2, w // 2
    y, x = np.indices(psd2d.shape)
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    r = r.astype(int)

    # Average over rings of equal radius
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile[:h//2]


def plot_test_sample(model, dataset, sample_idx, save_path):
    """Inputs u(t1), u(t2), true and predicted velocity field for one test sample."""
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
    """Relative L1 and L2 error of the predicted velocity field for every test sample, grouped by dt."""
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
    """Print and save (CSV + bar chart) the mean errors per dt and overall."""
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
    """Load a trained inverse CNO run and the inverse test dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    folder = str(paths.INVERSE_CHECKPOINT_DIR / model_name)
    ckpt_path = f"{folder}/{checkpoint_name}"
    data_path = str(paths.DATA_DIR)
    
    # Run properties and architecture
    files = [f"{folder}/inv_training_properties.txt", f"{folder}/inv_net_architecture.txt"]
    train_prop = pd.read_csv(files[0], header=None, index_col=0).squeeze().to_dict()
    net_arch = pd.read_csv(files[1], header=None, index_col=0).squeeze().to_dict()
    
    for d in [train_prop, net_arch]:
        for k, v in d.items(): d[k] = safe_eval(v)

    net_arch["batch_norm"] = net_arch.get("batch_norm") in [1, True, 'True']
    net_arch["is_att"] = net_arch.get("is_att", False) in [1, True, 'True']

    # Experiment configuration
    loader_dict = _load_inverse_dict(files=files, steps=15, which_example='wave_gauss', is_masked=False)
    loader_dict['data_path'] = data_path
    
    _allowed = list(range(0 if loader_dict.get("include_zero") else 1, int(train_prop.get("time_steps", 15)) + 1))
    loader_dict['allowed_transitions'] = _allowed

    # Model
    print(f"Initializing model on {device}...")
    model = InverseCNO.load_from_checkpoint(
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

    # Test dataset
    test_dataset = InverseWaveGaussians(
        which="test", resolution=128, num_trajectories=num_traj, 
        data_path=data_path, max_num_time_steps=15, 
        time_step_size=1, allowed_transitions=loader_dict['allowed_transitions']
    )

    return model, test_dataset

def plot_absolute_error_diagnostic(model, dataset, sample_idx, save_path):
    """True and predicted velocity field (shared colour scale) and their absolute error."""
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
    
    cmin = min(target_np.min(), pred_np.min())
    cmax = max(target_np.max(), pred_np.max())
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = ax[0].imshow(target_np, cmap='jet', vmin=cmin, vmax=cmax)
    ax[0].set_title("Ground Truth $c$")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    
    im1 = ax[1].imshow(pred_np, cmap='jet', vmin=cmin, vmax=cmax)
    ax[1].set_title(r"Predicted $\hat{c}$")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    
    # The error map has its own colour scale
    im2 = ax[2].imshow(abs_error, cmap='inferno')
    ax[2].set_title(r"Absolute Error $|c - \hat{c}|$")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Diagnostic error plot saved to: {save_path}")

def plot_spectral_analysis(model, dataset, device, output_dir, num_samples=120):
    """Radially averaged power spectrum of true vs. predicted velocity fields (one sample per test trajectory)."""
    model.eval()
    all_raps_true = []
    all_raps_pred = []
    
    print(f"Running spectral analysis on {num_samples} samples...")
    
    with torch.no_grad():
        # One sample per test trajectory: the snapshot pair (t = 0, t = 9)
        for i in tqdm(range(num_samples)):
            idx = i * 120 + 8
            time_val, input_tensor, target_c = dataset[idx]
            
            time_tensor = torch.tensor([time_val], dtype=torch.float32).to(device)
            input_batch = input_tensor.unsqueeze(0).to(device)
            prediction = model(input_batch, time_tensor)
            
            true_img = target_c[0].cpu().numpy()
            pred_img = prediction[0, 0].cpu().numpy()

            all_raps_true.append(_radial_power_spectrum(true_img))
            all_raps_pred.append(_radial_power_spectrum(pred_img))

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
    plt.title(r"Radial Averaged Power Spectrum: $c$ vs $\hat{c}$")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_path = f"{output_dir}/spectral_analysis_raps.pdf"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Spectral analysis plot saved to: {save_path}")

def main(model_name, checkpoint_name, num_traj):
    """Run all inverse-problem diagnostics for one trained run."""
    model, test_dataset = make_test_dataset_from_ckpt(model_name=model_name, checkpoint_name=checkpoint_name, num_traj=num_traj)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    output_dir = str(paths.RESULTS_DIR / "inverse" / model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nInverse CNO diagnostics")
    
    # Spectral analysis (radially averaged power spectrum) across all test trajectories
    plot_spectral_analysis(model, test_dataset, device, output_dir, num_samples=num_traj)
    
    # Representative samples: predictions and absolute-error maps for time gaps dt = 1 and 9
    print("\nGenerating representative plots...")
    for traj in range(5):
        for dt in [1, 9]:
            plot_absolute_error_diagnostic(model, test_dataset, traj * 120 + dt - 1, f"{output_dir}/diagnostic_abs_error_{traj}_dt_{dt}.pdf")
            plot_test_sample(model, test_dataset, traj * 120 + dt - 1, f"{output_dir}/test_trajectory_{traj}_dt_{dt}.pdf")

    # Relative L1/L2 errors per time gap dt (CSV + bar chart)
    metrics = calculate_test_metrics(model, test_dataset, device)
    summarize_and_save(metrics, output_dir)


if __name__ == "__main__":
    main(model_name=paths.INVERSE_RUN, checkpoint_name=paths.INVERSE_CKPT, num_traj=120)