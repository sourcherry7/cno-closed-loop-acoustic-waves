import os
import sys
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    folder = "/cluster/home/svisnjic/Thalassa/inv_results/checkpoints/model_1802_second"
    ckpt_path = f"{folder}/epoch=2-step=46080.ckpt"
    data_path = "/cluster/home/svisnjic/Thalassa/data"
    
    # # 1. Load Architecture
    # files = [f"{folder}/inv_training_properties.txt", f"{folder}/inv_net_architecture.txt"]
    # train_prop = pd.read_csv(files[0], header=None, index_col=0).squeeze().to_dict()
    # net_arch = pd.read_csv(files[1], header=None, index_col=0).squeeze().to_dict()
    
    # for d in [train_prop, net_arch]:
    #     for k, v in d.items(): d[k] = safe_eval(v)

    # net_arch["batch_norm"] = net_arch.get("batch_norm") in [1, True, 'True']
    # net_arch["is_att"] = net_arch.get("is_att", False) in [1, True, 'True']

    # # 2. Setup Loader Dictionary
    # loader_dict = _load_inverse_dict(files=files, steps=15, which_example='wave_gauss', is_masked=False)
    # loader_dict['data_path'] = data_path
    
    # _allowed = list(range(0 if loader_dict.get("include_zero") else 1, int(train_prop.get("time_steps", 15)) + 1))
    # loader_dict['allowed_transitions'] = _allowed

    # # 3. Model Initialization (RESTORED MISSING CONDITIONAL PARAMS)
    # print(f"Initializing model on {device}...")
    # model = CNO_time.load_from_checkpoint(
    #     ckpt_path,
    #     map_location=device,
    #     in_dim=loader_dict["in_dim"],
    #     out_dim=loader_dict.get("out_dim", 1),
    #     in_size=net_arch.get("in_size", 128),
    #     N_layers=net_arch.get("N_layers"),
    #     channel_multiplier=net_arch.get("channel_multiplier"),
    #     N_res=net_arch.get("N_res"),
    #     N_res_neck=net_arch.get("N_res_neck"),
    #     batch_norm=net_arch.get("batch_norm"),
    #     is_time=net_arch.get("is_time", 1),
    #     is_att=net_arch.get("is_att"),
    #     activation=net_arch.get("activation", "cno_lrelu_torch"),
    #     nl_dim=net_arch.get("nl_dim"),
    #     time_steps=loader_dict["time_steps"],
    #     time_input=loader_dict.get('time_input', True),
    #     loader_dictionary=loader_dict,
    #     strict=False, 
    #     allowed_transitions=loader_dict['allowed_transitions']
    # ).to(device)

    # # 4. Dataset Loading
    # test_dataset = InverseWaveGaussians(
    #     which="test", resolution=128, num_trajectories=240, 
    #     data_path=data_path, max_num_time_steps=15, 
    #     time_step_size=1, allowed_transitions=loader_dict['allowed_transitions']
    # )
    
    # 5. Run Testing
    output_dir = "/cluster/home/svisnjic/Thalassa/inv_results/testing_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    # print("\nGenerating representative plots...")
    
    # # for traj in range(5):
    # #     for dt in [1, 9]:
    # #         plot_test_sample(model, test_dataset, traj * 120 + dt - 1, f"{output_dir}/test_trajectory_{traj}_dt_{dt}.pdf")
    # #         print(f"Plotted trajctory {traj} at time pair (0, {dt}) and saved fig to: {output_dir}/test_trajectory_{traj}_dt_{dt}.pdf")

    df = pd.read_csv(f'{output_dir}/test_metrics_summary.csv')
    df = df.iloc[:-1]
    plt.figure(figsize=(6, 3.5))
    plt.bar(df["dt"], df["rel_l2"], color='#f28e2b')
    plt.xlabel("Time Step Gap (dt)")
    plt.ylabel("Mean Relative L2 Error")
    plt.title("Error vs Snapshot Separation")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(pad=1.1)
    plt.savefig(f"{output_dir}/error_by_dt.pdf")
    plt.close()
    print(f'Saved error graph to {output_dir}/error_by_dt.pdf')

    # metrics = calculate_test_metrics(model, test_dataset, device)
    # summarize_and_save(metrics, output_dir)

if __name__ == "__main__":
    main()