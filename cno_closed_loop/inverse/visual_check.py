"""Training-time visual check for the inverse CNO.

Saves u(t1), u(t2), their difference, the true velocity field and the predicted
one for a fixed sample, so reconstruction progress can be followed per epoch.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_reconstruction_sample(model, dataset, epoch, save_folder, sample_idx=4):
    """Save inputs, their difference, and true vs. predicted velocity field (same colour scale) for one sample."""
    model.eval()
    device = next(model.parameters()).device
    
    time_val, input_tensor, target_c = dataset[sample_idx]
    time_tensor = torch.tensor([time_val]).type(torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch, time_tensor)
    
    # Colour scales from the ground truth (c) and the first snapshot (u)
    target_np = target_c[0].cpu().numpy()
    c_min = np.min(target_np)
    c_max = np.max(target_np)

    u_min_t1 = np.min(input_tensor[0].cpu().numpy())
    u_max_t1 = np.max(input_tensor[0].cpu().numpy())
    
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    dt_actual = time_val * 15.0
    
    ax[0].imshow(input_tensor[0].cpu().numpy(), vmin=u_min_t1, vmax=u_max_t1)
    ax[0].set_title("Input: u(t1)")
    
    ax[1].imshow(input_tensor[1].cpu().numpy(), vmin=u_min_t1, vmax=u_max_t1)
    ax[1].set_title(f"Input: u(t2) (dt={dt_actual:.1f})")
    
    # Change between the two snapshots
    diff = (input_tensor[1] - input_tensor[0]).cpu().numpy()
    ax[2].imshow(diff, cmap='RdBu')
    ax[2].set_title("Motion (u2 - u1)")
    
    # True vs. predicted velocity field on the same scale
    ax[3].imshow(target_np, vmin=c_min, vmax=c_max)
    ax[3].set_title("Target: True Velocity (c)")
    
    ax[4].imshow(prediction[0, 0].cpu().numpy(), vmin=c_min, vmax=c_max)
    ax[4].set_title("Prediction: Recovered (c)")
    
    save_path = os.path.join(save_folder, f"epoch_{epoch}_reconstruction.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
