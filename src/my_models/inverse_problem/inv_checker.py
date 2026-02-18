import os
import torch
import matplotlib.pyplot as plt
from inv_CNO_timeModule_CIN import CNO_time # Ensure this path is correct
from inv_utils import _load_inverse_dict

def verify_inverse_training(checkpoint_path, loader_dict):
    # 1. Load the model from the checkpoint
    # Note: Use map_location='cpu' if you aren't on a GPU node
    model = CNO_time.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device('cpu'),  # Crucial since your terminal shows no CUDA runtime
            # Manually provide the missing positional arguments
            in_dim=loader_dict["in_dim"],
            in_size=128,                       # Or loader_dict["in_size"] if available
            N_layers=2,                        # Match the architecture you used for the dummy
            channel_multiplier = 8,
            N_res = 1,
            N_res_neck = 1,
            time_input = loader_dict['time_input'],
            loader_dictionary=loader_dict,     # Keep this for internal logic
            strict=False,                       # Useful if some weights have slight name mismatches
            allowed_transitions = loader_dict['allowed_transitions']
        )
    model.eval()
    
    # 2. Get a single batch from your Inverse Dataset
    from inv_wave_loader import InverseWaveGaussians # Your custom class
    test_dataset = InverseWaveGaussians(
        which="val", resolution=128, num_trajectories=1, 
        data_path=loader_dict['data_path'], max_num_time_steps=15, time_step_size=1, allowed_transitions=loader_dict['allowed_transitions']
    )
    
    
    # 1. Grab a sample and perform inference
    time_val, input_tensor, target_c = test_dataset[5]
    dt_actual = time_val * 15.0 # Un-normalize for the title
    
    with torch.no_grad():
        time_tensor = torch.tensor([time_val]).type(torch.float32)
        prediction = model(input_tensor.unsqueeze(0), time_tensor)

    # 2. Expanded Plotting (5 Panels)
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    
    # Input 1: u(t1)
    ax[0].imshow(input_tensor[0].cpu().numpy())
    ax[0].set_title(f"Input 1: u(t1)")
    
    # Input 2: u(t2)
    ax[1].imshow(input_tensor[1].cpu().numpy())
    ax[1].set_title(f"Input 2: u(t2)\ndt = {dt_actual:.1f}")
    
    # Input 3: Difference (to see the wave motion)
    diff = input_tensor[1] - input_tensor[0]
    ax[2].imshow(diff.cpu().numpy(), cmap='RdBu')
    ax[2].set_title("Motion (u2 - u1)")
    
    # Target: Ground Truth c
    ax[3].imshow(target_c[0].cpu().numpy())
    ax[3].set_title("Target: True Velocity (c)")
    
    # Prediction: Recovered c
    ax[4].imshow(prediction[0, 0].cpu().numpy())
    ax[4].set_title("Prediction: Recovered (c)")
    
    plt.tight_layout()
    plt.savefig("inv_verification.png")
    plt.close()
    print(f"Verification Plot saved. If Target is 'c', the logic is correct.")


import numpy as np

def save_reconstruction_sample(model, dataset, epoch, save_folder, sample_idx=5):
    """
    Visualizes the progress of the inverse problem with fixed color scaling.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Grab sample and prepare tensors
    time_val, input_tensor, target_c = dataset[sample_idx]
    time_tensor = torch.tensor([time_val]).type(torch.float32).to(device)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # 2. Inference
    with torch.no_grad():
        prediction = model(input_batch, time_tensor)
    
    # 3. Scale Extraction
    # Get min/max from the ground truth to use as the fixed scale
    target_np = target_c[0].cpu().numpy()
    c_min = np.min(target_np)
    c_max = np.max(target_np)
    
    # 4. Plotting
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    dt_actual = time_val * 15.0
    
    # Wave Snapshots
    ax[0].imshow(input_tensor[0].cpu().numpy())
    ax[0].set_title("Input: u(t1)")
    
    ax[1].imshow(input_tensor[1].cpu().numpy())
    ax[1].set_title(f"Input: u(t2) (dt={dt_actual:.1f})")
    
    # Difference (Motion) - often useful with RdBu or seismic cmaps
    diff = (input_tensor[1] - input_tensor[0]).cpu().numpy()
    ax[2].imshow(diff, cmap='RdBu')
    ax[2].set_title("Motion (u2 - u1)")
    
    # Target vs Prediction with SAME SCALE
    ax[3].imshow(target_np, vmin=c_min, vmax=c_max)
    ax[3].set_title("Target: True Velocity (c)")
    
    ax[4].imshow(prediction[0, 0].cpu().numpy(), vmin=c_min, vmax=c_max)
    ax[4].set_title("Prediction: Recovered (c)")
    
    # 5. Finalize
    save_path = os.path.join(save_folder, f"epoch_{epoch}_reconstruction.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Point this to one of your .ckpt files from the image
    ckpt = "/cluster/home/svisnjic/Thalassa/inv_results/checkpoints/model_1602/epoch=1-step=12.ckpt"
    # Use the loader_dict you created in your main script
    folder = "/cluster/home/svisnjic/Thalassa/inv_results/checkpoints" # "--- PROVIDE THE FOLDER TO SAVE THE MODEL ----" 
    files = [folder + '/inv_training_properties.txt', folder + '/inv_net_architecture.txt']
    loader_dict = _load_inverse_dict(files=files, steps =15, which_example='wave_gauss', is_masked = False)
    loader_dict['data_path'] = "/cluster/home/svisnjic/Thalassa/data"
    # training properties snippet
    training_properties = {}
    training_properties['allowed'] = 'all'
    # Which transitions during the training are allowed?
    _allowed = []
    if "include_zero" in loader_dict and loader_dict["include_zero"]:
        start_t = 0
    else:
        start_t = 1
    if training_properties['allowed'] == 'all':
        for t in range(start_t,loader_dict["time_steps"]+1):
            _allowed.append(t)
    elif training_properties['allowed']  == "one2all":
        _allowed = None
    elif training_properties['allowed'] == 'one':
        _allowed = [1]
    print("allowed transitions: ", loader_dict["allowed_transitions"])
    print("loader dict: ", loader_dict)
    verify_inverse_training(ckpt, loader_dict)