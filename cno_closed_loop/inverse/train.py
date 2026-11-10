# Training script for the inverse CNO: (u(t1), u(t2), dt) -> velocity field c.
#
# Adapted from TrainCNO_time_L.py of camlab-ethz/ConvolutionalNeuralOperator
# (MIT License), commit 75a3b9b. Changes: inverse-problem model, data and
# configuration (InverseCNO, inverse dataset/config), hyperparameters, a per-epoch
# visual-check callback, and run property files moved into the run folder.
# Run from the repository root:  python -m cno_closed_loop.inverse.train

from pytorch_lightning import Trainer
from cno_closed_loop import paths
from cno_closed_loop.inverse.model import InverseCNO
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import copy
import json
import os
import sys
import shutil

import pandas as pd
import torch


from pytorch_lightning.callbacks import Callback
from cno_closed_loop.inverse.visual_check import save_reconstruction_sample


class VisualVerificationCallback(Callback):
    def __init__(self, save_dir, frequency=10):
        super().__init__()
        self.save_dir = save_dir
        self.frequency = frequency
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.frequency == 0:
            # --- PROTECTIVE WRAPPER ---
            try:
                dataset = trainer.train_dataloader.dataset
                save_reconstruction_sample(
                    model=pl_module, 
                    dataset=dataset, 
                    epoch=trainer.current_epoch, 
                    save_folder=self.save_dir
                )
                print(f"Successfully saved visual for epoch {trainer.current_epoch}")
            except Exception as e:
                # If something goes wrong (OOM, device error, etc.), 
                # we just print it and keep training.
                print(f"\n[Visuals Warning] Plotting failed at epoch {trainer.current_epoch}: {e}")

if len(sys.argv) <= 2:
    
    training_properties = {
        "learning_rate": 0.0002,      # Slightly lower for stability with larger model capacity
        "weight_decay": 1e-4,
        "scheduler_step": 3,          # Adjusted for longer epoch count
        "scheduler_gamma": 0.9,
        "epochs": 30,                 # Inverse problems require more time to resolve structural maps
        "batch_size": 32,
        "time_steps": 15,              # Use the full simulation depth
        "dt": 1,                       
        "training_samples": 2048,      # Critical: Increased to capture diverse c-field patterns -> 2048
        "time_input": True,            
        "allowed": 'all',              # Enables All-to-All snapshot pairs (physics augmentation)
        "cluster": True,
    }

    model_architecture_ = {
        "N_layers": 4,                 # Deep U-Net structure for global context on 128x128 grid
        "channel_multiplier": 64,      # Doubled capacity to store complex hidden features
        "N_res": 8,                    # Deep residual processing in blocks
        "N_res_neck" : 8,              # Strong bottleneck for feature extraction
        
        "batch_norm": 1,               
        "is_time": 1,                  # Conditional Norms enabled for CIN logic
        "nl_dim": "23",                # '23' for Instance Norm-based conditioning
        
        "in_size": 128,                
        "activation": 'cno_lrelu_torch',
        
        "is_att": False,               
        "patch_size" : 1, 
        "dim_multiplier" : 1,
        "depth" : 2,
        "heads" : 4,                   # only used if is_att is True
        "dim_head_multiplier" : 0.5,
        "mlp_dim_multiplier" : 1.0,
        "emb_dropout" : 0.2,           # Small dropout to prevent overfitting on specific wave shapes
    }

    # AVAILABLE EXPERIMENTS:
    # "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear
    # "ns_pwc_t:
    # "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
    # "rich_mesh", "rayl_tayl" "kolmogorov"
    # "wave_seismic", "wave_gauss", "allen_cahn"
    # "airfoil", "poisson_gauss", "helmholtz"
    
    # FOR PRETRAINING CNO-FM: which_example = "eul_ns_mix1"
    
    # WHAT IS THE EXPERIMENT?
    which_example = "wave_gauss"
    
    folder = str(paths.INVERSE_CHECKPOINT_DIR)
    
else:
    raise ValueError("To many args")
    
    

#---------------------------------------------------------    
cluster = True  # We always tun on cluster

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
if not cluster: # Set the defaulf folder
    folder = "--- PROVIDE THE FOLDER TO SAVE THE MODEL (no cluster) ----"  

if not os.path.exists(folder):
    os.makedirs(folder)

df = pd.DataFrame.from_dict([training_properties]).T
df.to_csv(folder + '/inv_training_properties.txt', header=False, index=True, mode='w')
df = pd.DataFrame.from_dict([model_architecture_]).T
df.to_csv(folder + '/inv_net_architecture.txt', header=False, index=True, mode='w')

#---------------------------------------------------------
# Load parameters related to the specific experiment -- "DataLoaders/all_experiments.json"

# Inverse-problem experiment configuration (replaces DataLoaders/all_experiments.json)
from cno_closed_loop.inverse.config import _load_inverse_dict

if which_example == 'wave_gauss':
    files = [folder + '/inv_training_properties.txt', folder + '/inv_net_architecture.txt']
else:
    raise ValueError("Only the wave_gauss experiment is implemented for the inverse problem")
loader_dict = _load_inverse_dict(files=files, which_example=which_example)



# loader_dict: INFORMATION ABOUT THE EXPERIMENT, TRAINING, etc -- VERY IMPORTANT!
loader_dict["which"] = which_example
loader_dict["time_input"] = training_properties["time_input"]
loader_dict["cluster"] = training_properties["cluster"]
loader_dict["num_samples"] = training_properties["training_samples"]
loader_dict["dt"] = training_properties["dt"]
loader_dict["time_steps"] = training_properties["time_steps"]

#---------------------------------------------------------
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
loader_dict["allowed_tran"] = _allowed

#---------------------------------------------------------
# Initialize CNO
model   = InverseCNO(in_dim =  loader_dict["in_dim"],               
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

ver = "_0203_new_data" # Just a random string to be added to the model name

checkpoint_callback = ModelCheckpoint(dirpath = folder+"/model"+str(ver), monitor='mean_val_l')
early_stop_callback = EarlyStopping(monitor="mean_val_l", patience=10)

# Initialize the callback
visual_callback = VisualVerificationCallback(
    save_dir=folder + "/visuals" + str(ver), 
    frequency=1
)

logger = TensorBoardLogger(save_dir=folder, version=ver, name="logs")
trainer = Trainer(devices = -1,
                max_epochs = training_properties["epochs"],
                callbacks = [checkpoint_callback,early_stop_callback, visual_callback],
                strategy="ddp_find_unused_parameters_true", #IMPORTANT!!!
                logger=logger)
trainer.fit(model)
trainer.validate(model)

#---------------------------------------------------------

# Move property files to model dir to avoid inproper overwriting


files_to_move = files
dest_dir = folder+"/model"+str(ver)

os.makedirs(dest_dir, exist_ok=True)

for movable_file in files_to_move:
    if os.path.exists(movable_file):
        shutil.move(movable_file, dest_dir)
    else:
        print(f"File not found: {movable_file}")