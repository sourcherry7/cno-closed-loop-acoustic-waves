"""Loading of a trained forward CNO, its configuration and Wave-Gauss test samples.

Wraps the CNO configuration/model/data helpers from external/CNO
(test_and_fine_tune_utils.test_utils). Running the module performs a quick
check of the trained forward model:  python -m cno_closed_loop.forward.model_loader
"""

import os
import torch
import numpy as np

from cno_closed_loop import paths

# CNO utilities (external/CNO)
from test_and_fine_tune_utils.test_utils import _load_dict, _initialize_model, _load_data

class WaveLoader:
    """Configuration, model and test data of a trained forward CNO run."""

    def __init__(self, train_file, net_file, ckpt_path, which_name="wave_gauss"):
        self.train_file = train_file
        self.net_file = net_file
        self.ckpt_path = ckpt_path
        self.which_name = which_name
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loader_dict = None

    def _ensure_config(self):
        """Read the run's training/architecture files into the CNO configuration dict (once)."""
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
        """Build the CNO from the configuration and load the checkpoint weights."""
        self._ensure_config()
        print("Initializing model architecture...")
        model = _initialize_model(self.loader_dict)
        
        print(f"Loading weights from {self.ckpt_path}")
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)

        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def load_data_sample(self, batch_size=1):
        """First test batch (time, input, target), with the input fixed at t = 0."""
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

    def get_sample_by_index(self, idx):
        """Test sample `idx` as (time, input, velocity field in normalized units)."""
        self._ensure_config()
        from DataLoaders.CNO_TimeLoaders import WaveGaussians
        
        if not hasattr(self, '_index_dataset'):
            step_max = self.loader_dict.get("step_max", 15)
            
            common_args = {
                "max_num_time_steps": step_max,
                "time_step_size": 1,
                "resolution": 128,
                "num_trajectories": -1, 
                "time_input": True,
                "which": "test",
            }
            data_dir = str(paths.DATA_DIR)
            
            print("Loading the Wave-Gauss test set...")
            # WaveGaussians (CNO fork, closed-loop-waves branch) reads new_wave_gauss_data.h5
            self._index_dataset = WaveGaussians(data_path=data_dir, **common_args)
        
        raw_sample = self._index_dataset[idx]
        t_scalar = raw_sample[0].unsqueeze(0).to(self.device)     
        x_input = raw_sample[1].unsqueeze(0).to(self.device)      
        
        # Input channels are [u, c, dt]
        c_velocity = x_input.squeeze().cpu().numpy()[1]          
        
        return t_scalar, x_input, c_velocity


# Quick check of the trained forward model on one test sample
if __name__ == "__main__":
    base_dir = str(paths.FORWARD_CHECKPOINT_DIR / paths.FORWARD_RUN)
    loader = WaveLoader(os.path.join(base_dir, "training_properties.txt"),
                        os.path.join(base_dir, "net_architecture.txt"),
                        os.path.join(base_dir, paths.FORWARD_CKPT), which_name="wave_gauss")
    model = loader.load_model()
    print(f"Model on {loader.device}: {sum(p.numel() for p in model.parameters()):,} parameters")

    t, x, y_true = loader.load_data_sample()          # input channels [u, c, dt], targets [u, c]
    with torch.no_grad():
        y_pred = model(x, t)
    y_pred_np = y_pred.squeeze(0).cpu().numpy()
    y_true_np = y_true.squeeze(0).cpu().numpy()

    rel_l1 = np.sum(np.abs(y_pred_np[0] - y_true_np[0])) / (np.sum(np.abs(y_true_np[0])) + 1e-8) * 100
    print(f"One-step relative L1 error of u: {rel_l1:.2f}%")
    if y_pred_np.shape[0] > 1:
        # The velocity field is static, so the predicted c should stay equal to the input c
        c_in = x.squeeze(0).cpu().numpy()[1]
        print(f"Mean |c_in - c_pred| (normalized units): {np.mean(np.abs(c_in - y_pred_np[1])):.5f}")
