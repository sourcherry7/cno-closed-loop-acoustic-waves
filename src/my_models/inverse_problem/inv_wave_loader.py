import os
import sys
import h5py
import json
import torch
import numpy as np


PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
EXTERNAL_CNO_DIR = f"{PROJECT_ROOT}/src/external/CNO/CNO2d_time_dependent_&_foundation_model"
sys.path.append(EXTERNAL_CNO_DIR)

from DataLoaders.CNO_TimeLoaders import BaseTimeDataset


class InverseWaveGaussians(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_path = self.data_path + "/new_wave_gauss_data.h5"
        self.reader = h5py.File(data_path, "r")

        tot_n_samples = self.reader['u'].shape[0]

        train_val_test_scale_factor = tot_n_samples / 10512.0
        
        self.N_max = tot_n_samples
        self.N_val = int(60 * train_val_test_scale_factor)
        self.N_test = int(240 * train_val_test_scale_factor)
        self.resolution = self.reader['u'].shape[-1]

        NORM_JSON_PATH = self.data_path + "/new_normalization_stats.json"
        with open(NORM_JSON_PATH, "r") as f:
            json_constants = json.load(f)


        self.constants = {
            "mean": json_constants['mean'], # 0.0334376316,
            "std": json_constants['std'], # 0.1171879068,
            "mean_c": json_constants['mean_c'], # 2618.4593933,
            "std_c": json_constants['std_c'], # 601.51658913,
            "time": float(json_constants['time_steps']) # 15.0,
        }

        self.input_dim = 2 # now it's ut1 and ut2 instead of u, c. dt will be appended later
        self.label_description = "[c]"
        self.time_input = True

        assert self.max_num_time_steps <= json_constants['time_steps']
        assert self.max_num_time_steps * self.time_step_size <= 15


        self.post_init()


    def __getitem__(self, idx):
        i = idx // self.multiplier # select trajectory
        _idx = idx - i * self.multiplier # select time_pair

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
            t = t2 - t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
            t = t2 - t1
        time = t / self.constants["time"]

        inputs_t1 = (
            torch.from_numpy(self.reader['u'][i + self.start, t1])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs_t2 = (
            torch.from_numpy(self.reader['u'][i + self.start, t2])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        labels = (
            torch.from_numpy(self.reader["c"][i + self.start])
            .type(torch.float32)
            .reshape(1, self.resolution, self.resolution)
        )

        inputs_t1 = (inputs_t1 - self.constants['mean']) / self.constants['std']
        inputs_t2 = (inputs_t2 - self.constants['mean']) / self.constants['std']
        labels = (labels - self.constants['mean_c']) / self.constants['std_c']

        inputs = torch.cat([inputs_t1, inputs_t2], dim=0)

        inputs_t = torch.ones(1, self.resolution, self.resolution).type(torch.float32)*time
        inputs = torch.cat((inputs, inputs_t), 0)

        return time, inputs, labels



    
if __name__ == "__main__":

    ds = InverseWaveGaussians(
        which="train", 
        resolution=128, 
        num_trajectories=10, 
        data_path="/cluster/home/svisnjic/Thalassa/data",
        time_input=True,
        max_num_time_steps=15,
        time_step_size=1,
    )

    print(f"total loaded samples: {len(ds)}")

    # 2. Test a single item retrieval
    time_val, input_tensor, label_tensor = ds[0]

    print("\n--- Single Item Check ---")
    print(f"Time (normalized): {time_val:.4f}")
    print(f"Input Shape:  {input_tensor.shape}")  # Expected: [3, 128, 128]
    print(f"Label Shape:  {label_tensor.shape}")  # Expected: [1, 128, 128]
    print(f"Input Range:  {input_tensor.min():.2f} to {input_tensor.max():.2f}")
    print(f"Label Range:  {label_tensor.min():.2f} to {label_tensor.max():.2f}")

    # 3. Test the DataLoader (Batching)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    batch_time, batch_in, batch_label = next(iter(loader))
    print("\n--- Batch Check ---")
    print(f"Batch Input Shape: {batch_in.shape}") # Expected: [4, 3, 128, 128]
    print(f"Batch Label Shape: {batch_label.shape}") # Expected: [4, 1, 128, 128]
    
    print("\nSanity Check Passed! You are ready for the CNO.")