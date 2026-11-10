# Training script for the forward (time-dependent) CNO on the Wave-Gauss data.
#
# Adapted from TrainCNO_time_L.py of camlab-ethz/ConvolutionalNeuralOperator
# (MIT License), commit 75a3b9b. Changes: Wave-Gauss configuration and
# hyperparameters, a callback that plots a 15-step autoregressive rollout against
# the ground truth after each validation epoch, optional resuming, and run
# property files moved into the run folder.
# Run from the repository root:  python -m cno_closed_loop.forward.train

import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from cno_closed_loop import paths  # also makes the CNO code (external/CNO) importable
from CNO_timeModule_CIN import CNO_time


class RolloutVisualCallback(Callback):
    """After validation, plot a 15-step autoregressive rollout of the first validation
    trajectory against the ground truth (saved as rollout_epoch_<n>.pdf)."""

    def __init__(self, save_dir, frequency=10):
        super().__init__()
        self.save_dir = save_dir
        self.frequency = frequency
        os.makedirs(save_dir, exist_ok=True)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.frequency == 0 and trainer.current_epoch > 0:
            pl_module.eval()
            try:
                # First validation trajectory, all 16 frames, in physical units
                val_dataset = trainer.val_dataloaders[0].dataset
                idx = val_dataset.start 
                u_gt_phys = val_dataset.reader['u'][idx] # [16, 128, 128]
                c_phys = val_dataset.reader['c'][idx]    # [128, 128]
                
                # Normalized model inputs
                stats = val_dataset.constants
                device = pl_module.device
                u_curr = (torch.from_numpy(u_gt_phys[0]).float().to(device) - stats["mean"]) / stats["std"]
                u_curr = u_curr.view(1, 1, 128, 128)
                
                c_norm = (torch.from_numpy(c_phys).float().to(device) - stats["mean_c"]) / stats["std_c"]
                c_norm = c_norm.view(1, 1, 128, 128)
                
                # One time step in normalized time (dt = 1/15)
                dt_val = 1.0 / stats["time"]
                t_scalar = torch.tensor([[dt_val]], device=device)
                
                # Time channel as third input (if enabled)
                if val_dataset.time_input:
                    t_channel = torch.ones_like(c_norm) * dt_val

                # Autoregressive rollout
                preds_phys = [u_gt_phys[0]]
                
                for _ in range(1, 16):
                    # Input channels: [u, c, dt] (or [u, c])
                    if val_dataset.time_input:
                        x_in = torch.cat([u_curr, c_norm, t_channel], dim=1)
                    else:
                        x_in = torch.cat([u_curr, c_norm], dim=1)
                        
                    out = pl_module(x_in, t_scalar)
                    u_next_norm = out[:, 0:1] # wavefield channel
                    
                    u_phys = (u_next_norm.squeeze().cpu().numpy() * stats["std"]) + stats["mean"]
                    preds_phys.append(u_phys)
                    u_curr = u_next_norm

                preds_phys = np.array(preds_phys)

                # Ground truth (rows 0-3) vs. CNO rollout (rows 4-7)
                u_min = min(u_gt_phys.min(), preds_phys.min())
                u_max = max(u_gt_phys.max(), preds_phys.max())
                
                fig, axes = plt.subplots(8, 4, figsize=(20, 32))
                for i in range(16):
                    ax_gt = axes[i // 4, i % 4]
                    im_gt = ax_gt.imshow(u_gt_phys[i], vmin=u_min, vmax=u_max, cmap='jet')
                    ax_gt.set_title(f"GT u(t={i})")
                    ax_gt.axis('off')
                    
                    ax_pred = axes[(i // 4) + 4, i % 4]
                    im_pred = ax_pred.imshow(preds_phys[i], vmin=u_min, vmax=u_max, cmap='jet')
                    ax_pred.set_title(f"CNO u(t={i})")
                    ax_pred.axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(self.save_dir, f"rollout_epoch_{trainer.current_epoch}.pdf")
                plt.savefig(save_path)
                plt.close()
                print(f"\nSaved rollout plot to {save_path}")

            except Exception as e:
                # A failed plot must not stop the training run
                print(f"\nRollout plot failed (training continues): {e}")

if len(sys.argv) <= 2:
    
    training_properties = {
        "learning_rate": 0.0003, 
        "weight_decay": 1e-6,
        "scheduler_step": 3,
        "scheduler_gamma": 0.7,
        "epochs": 40,
        "batch_size": 32,         
        "time_steps": 15,         # time steps per trajectory
        "dt": 1,                  # spacing of the used snapshots (1 = every snapshot)
        "training_samples": 2048, # number of training trajectories
        "time_input": 1,          # add dt as an input channel
        "allowed": 'all',         # time pairs: 'all' (all-to-all), 'one2all' or 'one' (single steps)
        "cluster": True,          # upstream option, keep True
    }
    
    model_architecture_ = {
        "N_layers": 4,            # number of down-/upsampling blocks
        "channel_multiplier": 64, # channel width multiplier
        "N_res": 8,               # residual blocks per level
        "N_res_neck" : 8,         # residual blocks in the bottleneck
        
        "batch_norm": 1,          # plain batch norm (not used when is_time = 1)
        "is_time": 1,             # time-conditioned normalization
        "nl_dim": "23",           # normalization: '23' instance, '023' batch, '123' layer
        
        "in_size": 128,           # grid resolution
        "activation": 'cno_lrelu_torch',# cno_lrelu, cno_lrelu_torch or lrelu
        
        "is_att": False,          # attention in the bottleneck
        "patch_size" : 1,         # attention (ViT) parameters, used only if is_att
        "dim_multiplier" : 1,
        "depth" : 2,
        "heads" : 2,
        "dim_head_multiplier" : 0.5,
        "mlp_dim_multiplier" : 1.0,
        "emb_dropout" : 0.
        }
    
    which_example = "wave_gauss"
    new_which_example = "new_wave_gauss"  # all_experiments.json entry for the 5,256-trajectory HDF5 dataset
    
    folder = str(paths.FORWARD_CHECKPOINT_DIR)
    
else:
    raise ValueError("Too many arguments")

#---------------------------------------------------------
model_architecture_["batch_norm"] = True if model_architecture_["batch_norm"] in [1, True] else False
training_properties["time_input"] = True if training_properties["time_input"] in [1, True] else False
model_architecture_["is_att"]     = True if model_architecture_["is_att"]     in [1, True] else False
training_properties["cluster"]    = True if training_properties["cluster"]    in [1, True] else False

if model_architecture_["nl_dim"] in ["023"]:
    model_architecture_["nl_dim"] = [0,2,3]
elif model_architecture_["nl_dim"] in ["123"]:
    model_architecture_["nl_dim"] = [1,2,3]
elif model_architecture_["nl_dim"] in ["23"]:
    model_architecture_["nl_dim"] = [2,3]

#---------------------------------------------------------
if not os.path.exists(folder):
    os.makedirs(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/net_architecture.txt', header=False, index=True, mode='w')

#---------------------------------------------------------
# Experiment settings from the CNO code (DataLoaders/all_experiments.json)

Dict_EXP = json.load( open( paths.CNO_DIR / "DataLoaders" / "all_experiments.json" ))
if new_which_example in Dict_EXP:
    loader_dict =  Dict_EXP[new_which_example]
else:
    raise ValueError("Please specify different benchmark")

# Experiment and training settings used by the CNO module and its data loaders
loader_dict["which"] = which_example
loader_dict["time_input"] = training_properties["time_input"]
loader_dict["cluster"] = training_properties["cluster"]
loader_dict["num_samples"] = training_properties["training_samples"]
loader_dict["dt"] = training_properties["dt"]
loader_dict["time_steps"] = training_properties["time_steps"]

#---------------------------------------------------------
# Allowed time gaps between input and target during training
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
loader_dict["allowed_tran"] = _allowed

#---------------------------------------------------------
model   = CNO_time(in_dim =  loader_dict["in_dim"],               
                    in_size  = 128,                  
                    N_layers = model_architecture_["N_layers"],                
                    N_res =  model_architecture_["N_res"], 
                    N_res_neck = model_architecture_["N_res_neck"],           
                    channel_multiplier = model_architecture_["channel_multiplier"],  
                    batch_norm = model_architecture_["batch_norm"],    
                    out_dim = loader_dict["out_dim"],               
                    activation = model_architecture_["activation"], 
                    time_steps = loader_dict["time_steps"],
                    is_time = model_architecture_["is_time"],
                    nl_dim= model_architecture_["nl_dim"],

                    lr = training_properties["learning_rate"],
                    batch_size = training_properties["batch_size"],
                    weight_decay = training_properties["weight_decay"],
                    scheduler_step = training_properties["scheduler_step"],
                    scheduler_gamma = training_properties["scheduler_gamma"],

                    loader_dictionary = loader_dict,

                    is_att = model_architecture_["is_att"],
                    patch_size = model_architecture_["patch_size"],
                    dim_multiplier = model_architecture_["dim_multiplier"],
                    depth = model_architecture_["depth"],
                    heads = model_architecture_["heads"],
                    dim_head_multiplier = model_architecture_["dim_head_multiplier"],
                    mlp_dim_multiplier = model_architecture_["mlp_dim_multiplier"],
                    emb_dropout = model_architecture_["emb_dropout"])

#---------------------------------------------------------

ver = "_0903_newgauss" # run name suffix: checkpoints go to model<ver>/

visual_callback = RolloutVisualCallback(
    save_dir=folder + "/visuals" + str(ver), 
    frequency=1  # save every epoch
)

checkpoint_callback = ModelCheckpoint(dirpath = folder+"/model"+str(ver), monitor='mean_val_l')
early_part = 10
early_stop_callback = EarlyStopping(monitor="mean_val_l", patience=training_properties["epochs"]//early_part)

logger = TensorBoardLogger(save_dir=folder, version=ver, name="logs")
trainer = Trainer(devices = -1,
                max_epochs = training_properties["epochs"],
                callbacks = [checkpoint_callback,early_stop_callback, visual_callback],
                strategy="ddp_find_unused_parameters_true", # some CNO parameters get no gradient
                logger=logger)

# Optionally resume an interrupted run
RESUME_CKPT = None  # e.g. "epoch=3-step=34816.ckpt" (a checkpoint inside the run folder)
checkpoint_path = os.path.join(folder, "model" + str(ver), RESUME_CKPT) if RESUME_CKPT else None

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Resuming from {checkpoint_path}")
    trainer.fit(model, ckpt_path=checkpoint_path)
else:
    print("Starting a new training run")
    trainer.fit(model)

trainer.validate(model)

dest_dir = folder + "/model" + str(ver)
os.makedirs(dest_dir, exist_ok=True)

# Keep the run's property files next to its checkpoints
for movable_file in ['/training_properties.txt', '/net_architecture.txt']:
    src = folder + movable_file
    if os.path.exists(src):
        shutil.move(src, dest_dir + movable_file)
#---------------------------------------------------------