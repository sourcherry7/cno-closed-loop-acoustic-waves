import os
import sys
import torch
import numpy as np
import json

# --- 1. SETUP PATHS ---
PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
EXTERNAL_CNO_DIR = f"{PROJECT_ROOT}/src/external/CNO/CNO2d_time_dependent_&_foundation_model"
sys.path.append(EXTERNAL_CNO_DIR)

# --- 2. IMPORT UTILS FROM REPO ---
from test_and_fine_tune_utils.test_utils import _load_dict, _initialize_model, _load_data

class WaveLoader:
    def __init__(self, train_file, net_file, ckpt_path, which_name="wave_gauss"):
        self.train_file = train_file
        self.net_file = net_file
        self.ckpt_path = ckpt_path
        self.which_name = which_name
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loader_dict = None

    def _ensure_config(self):
        if self.loader_dict is not None:
            return

        print(f"Loading configuration from {self.train_file}...")
        steps = 20 if "seismic" in self.which_name else 15

        self.loader_dict = _load_dict(
            files=[self.train_file, self.net_file], 
            which_example=self.which_name, 
            steps=steps
        )
        if "exp" not in self.loader_dict:
            self.loader_dict["exp"] = 1

    def load_model(self):
        self._ensure_config()
        print("Initializing model architecture...")
        model = _initialize_model(self.loader_dict)
        
        print(f"Loading weights from {self.ckpt_path}")
        try:
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint not found at {self.ckpt_path}")

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def load_data_sample(self, batch_size=1):
        self._ensure_config()
        print(f"Loading data loader (batch_size={batch_size})...")
        loader = _load_data(
            which=self.which_name, 
            loader_dict=self.loader_dict, 
            batch_size=batch_size, 
            which_loader="test",
            fix_input_to_time_step=0 
        )

        batch = next(iter(loader))
        
        t = batch[0].to(self.device) 
        x = batch[1].to(self.device) 
        y = batch[2].to(self.device) 
        
        return t, x, y

# --- UPDATE IN wave_loaders.py (get_sample_by_index) ---
    def get_sample_by_index(self, idx):
        self._ensure_config()
        from DataLoaders.CNO_TimeLoaders import WaveGaussians
        
        if not hasattr(self, '_index_dataset'):
            # Use the actual step_max from your new JSON/config
            step_max = self.loader_dict.get("step_max", 15)
            
            common_args = {
                "max_num_time_steps": step_max,
                "time_step_size": 1,
                "resolution": 128,
                "num_trajectories": -1, 
                "time_input": True, # Ensure time channel is active
                "which": "test",
            }
            # Use the PROJECT_ROOT to find the new data directory
            data_dir = os.path.join(PROJECT_ROOT, "data")
            
            print(f"Initializing Index Dataset for NEW GAUSS data...")
            # This triggers your updated WaveGaussians class which loads 'new_wave_gauss_data.h5'
            self._index_dataset = WaveGaussians(data_path=data_dir, **common_args)
        
        raw_sample = self._index_dataset[idx]
        t_scalar = raw_sample[0].unsqueeze(0).to(self.device)     
        x_input = raw_sample[1].unsqueeze(0).to(self.device)      
        
        # Velocity is index 1 in the [u, c, t] stack
        c_velocity = x_input.squeeze().cpu().numpy()[1]          
        
        return t_scalar, x_input, c_velocity

    # def get_sample_by_index(self, idx):
    #     """
    #     Get a specific sample from the dataset by index.
    #     Returns: (time_t, x_input, c_velocity)
    #     """
    #     self._ensure_config()
    #     from DataLoaders.CNO_TimeLoaders import WaveSeismic, WaveGaussians
        
    #     if not hasattr(self, '_index_dataset'):
    #         step_max = self.loader_dict.get("step_max", self.loader_dict.get("steps", 15))
            
    #         common_args = {
    #             "max_num_time_steps": step_max,
    #             "time_step_size": 1,
    #             "resolution": 128,
    #             "in_dist": True,
    #             "num_trajectories": -1, 
    #             "time_input": 1,
    #             "which": "test",
    #         }
    #         data_dir = f"{PROJECT_ROOT}/data"
            
    #         print(f"Initializing Index Dataset (loading all trajectories)...")
    #         if "seismic" in self.which_name:
    #             self._index_dataset = WaveSeismic(data_path=data_dir, **common_args)
    #         else:
    #             self._index_dataset = WaveGaussians(data_path=data_dir, **common_args)
        
    #     raw_sample = self._index_dataset[idx]
        
    #     t_scalar = raw_sample[0].unsqueeze(0).to(self.device)     # [1, 1]
    #     x_input = raw_sample[1].unsqueeze(0).to(self.device)      # [1, C, H, W]
        
    #     # Extract c (Velocity). 
    #     # Structure is [u, c, t]. C is at index 1.
    #     c_velocity = x_input.squeeze().cpu().numpy()[1]          # [H, W]
        
    #     return t_scalar, x_input, c_velocity

# --- FULL DIAGNOSTIC SUITE (FIXED) ---
if __name__ == "__main__":
    # 1. SETUP
    base_dir = f"{PROJECT_ROOT}/fwd_results/checkpoints/model_0903_newgauss"
    train_file = os.path.join(base_dir,  "training_properties.txt")
    net_file = os.path.join(base_dir, "net_architecture.txt")
    ckpt = os.path.join(base_dir, "epoch=20-step=182784.ckpt")

    loader = WaveLoader(train_file, net_file, ckpt, which_name="wave_gauss")
    
    print("\n" + "="*50)
    print("       WAVE MODEL DIAGNOSTIC       ")
    print("="*50)

    # 2. MODEL CHECK
    try:
        model = loader.load_model()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model Loaded successfully on {loader.device}")
        print(f"     Parameter Count: {param_count:,}")
    except Exception as e:
        print(f"[FAIL] Model Loading: {e}")
        sys.exit(1)

    # 3. DATA SHAPE CHECK
    t, x, y_true = loader.load_data_sample()
    print(f"\n[INFO] Data Shapes (Batch=1):")
    print(f"     Time (t):   {list(t.shape)}")
    print(f"     Input (x):  {list(x.shape)} -> [1, C, H, W]")
    print(f"     Target (y): {list(y_true.shape)}")

    # 4. COMPONENT STATISTICS (Normalization Check)
    x_np = x.squeeze(0).cpu().numpy()
    
    # Correct Channel Mapping for WaveGaussians: [u, c, t]
    if x_np.shape[0] == 3:
        u_curr = x_np[0]
        c_field = x_np[1] # Velocity is index 1
        t_field = x_np[2] # Time map is index 2
        print(f"\n[INFO] Channel Mapping Detected: [u, c, t]")
    else:
        # Fallback for other structures
        u_curr = x_np[0]
        c_field = x_np[1]
        t_field = None
    
    print(f"\n[INFO] Input Statistics:")
    print(f"     Wave (u) Range: [{u_curr.min():.3f}, {u_curr.max():.3f}] Mean: {u_curr.mean():.3f}")
    print(f"     Vel  (c) Range: [{c_field.min():.3f}, {c_field.max():.3f}] Mean: {c_field.mean():.3f}")
    if t_field is not None:
        print(f"     Time (t) Range: [{t_field.min():.3f}, {t_field.max():.3f}] (Should be constant)")
    
    # 5. SINGLE STEP PREDICTION ACCURACY
    print(f"\n[INFO] Running Single-Step Prediction (t -> t+1)...")
    with torch.no_grad():
        y_pred = model(x, t)
    
    y_pred_np = y_pred.squeeze(0).cpu().numpy()
    y_true_np = y_true.squeeze(0).cpu().numpy()
    
    # Analyze Wave Prediction (Channel 0)
    pred_wave = y_pred_np[0]
    true_wave = y_true_np[0]
    
    l1_diff = np.abs(pred_wave - true_wave)
    mae = np.mean(l1_diff)
    rel_l1 = np.sum(l1_diff) / (np.sum(np.abs(true_wave)) + 1e-8) * 100
    
    print(f"     Wave MAE:         {mae:.5f}")
    print(f"     Wave Relative L1: {rel_l1:.2f}%")
    
    if rel_l1 < 10.0:
        print(f"     [PASS] Model has learned physics! (<10% error)")
    else:
        print(f"     [FAIL/WARN] High one-step error.")

    # 6. VELOCITY RECONSTRUCTION CHECK (Channel 1)
    if y_pred_np.shape[0] > 1:
        pred_c = y_pred_np[1]
        
        # Check against Input C (Static field)
        drift = np.mean(np.abs(c_field - pred_c))
        print(f"\n[INFO] Velocity Drift Check:")
        print(f"     Input C vs Pred C MAE: {drift:.5f}")
        
        if drift > 0.05:
            print("     [WARN] Significant drift detected. Static Field Injection is required.")
        else:
            print("     [OK] Low drift.")
    
    print("\n" + "="*50)
    print("       DIAGNOSTIC COMPLETE         ")
    print("="*50)