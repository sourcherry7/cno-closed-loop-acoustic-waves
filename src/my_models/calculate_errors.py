import os
import sys
import torch
import torch.nn as nn
import numpy as np
import copy
import time

# --- 1. SETUP PATHS ---
PROJECT_ROOT = "/cluster/home/svisnjic/Thalassa"
EXTERNAL_CNO_DIR = f"{PROJECT_ROOT}/src/external/CNO/CNO2d_time_dependent_&_foundation_model"
sys.path.append(EXTERNAL_CNO_DIR)
sys.path.append(f"{PROJECT_ROOT}/src/my_models")

from test_and_fine_tune_utils.test_utils import _load_data
from wave_loaders import WaveLoader

def _write_in_file(file_name, test_dic, E, E_l1, times, pattern, separate_var=False, num_var=0):
    print(f"Writing results to {file_name}")
    with open(file_name, "w") as file:
        max_steps = int(float(test_dic["step_max"]))
        for t in range(len(times)):
            time_val = int(times[t]*max_steps)
            if not separate_var:
                errors = E[t]
                file.write(f"median {time_val},{np.median(errors)}\n")
                file.write(f"mean {time_val},{np.mean(errors)}\n")
                errors_l1 = E_l1[t]
                file.write(f"median_l1 {time_val},{np.median(errors_l1)}\n")
                file.write(f"mean_l1 {time_val},{np.mean(errors_l1)}\n")
            else:
                for i in range(num_var):
                    errors_i = E[i,t]
                    file.write(f"median {time_val}_{i},{np.median(errors_i)}\n")
                    file.write(f"mean {time_val}_{i},{np.mean(errors_i)}\n")
                    errors_i = E_l1[i,t]
                    file.write(f"median_l1 {time_val}_{i},{np.median(errors_i)}\n")
                    file.write(f"mean_l1 {time_val}_{i},{np.mean(errors_i)}\n")

def _test_pattern(model, testing_loader, batch_size=10, time_steps=5, dt=2, max_steps=10.0,
                  pattern=[1,1,1,1,1], device='cuda', folder="", separate_var=False,
                  separate_dim=[], is_masked=None, steady=False, is_airfoil=False):
    
    print(f"Testing Pattern: {pattern} | Batch Size: {batch_size}")
    delta = dt/max_steps
    T = []    
    
    if separate_var:
        len_sep = len(separate_dim)
        diff = [0]
        for val in separate_dim:
            diff.append(diff[-1] + val)
        E = np.zeros((len_sep, len(pattern), 0))
        E_l1 = np.zeros((len_sep, len(pattern), 0))
    else:
        E = np.zeros((len(pattern), 0))
        E_l1 = np.zeros((len(pattern), 0))
    
    std_out = 1.0
    std_inp = 1.0

    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(testing_loader):
            if is_masked:
                time_batch_, input_batch_, output_batch_, masked_dim = batch
            else:
                time_batch_, input_batch_, output_batch_ = batch
            
            if input_batch_.ndim == 3: input_batch_ = input_batch_.unsqueeze(0)
            if output_batch_.ndim == 3: output_batch_ = output_batch_.unsqueeze(0)
            if time_batch_.ndim == 1: time_batch_ = time_batch_.unsqueeze(1)

            t_batch = copy.deepcopy(time_batch_.to(device))
            input_batch = copy.deepcopy(input_batch_.to(device))
            output_batch = copy.deepcopy(output_batch_.to(device))
            
            acc_jump = 0
            dim = output_batch.shape[1] 

            for i, jump in enumerate(pattern):
                acc_jump += jump
                if step == 0: T.append(acc_jump * delta)
                
                # --- INITIALIZATION ---
                if i == 0:
                    inp = input_batch[0::time_steps] 
                    
                    # 1. Capture Static Velocity 'c' (Channel 1)
                    if separate_var and len(separate_dim) > 1:
                        c_idx = diff[1]
                        static_c_channel = inp[:, c_idx:c_idx+1, :, :].clone()
                    
                    # 2. Capture Static Step-Size 'dt' (Channel 2)
                    # We capture this once and reuse it to ensure the model sees constant dt.
                    if inp.shape[1] > 2:
                        static_dt_channel = inp[:, 2:3, :, :].clone()

                    zero_time_inp = torch.zeros_like(t_batch[0::time_steps]).to(device)
                    batch_curr_size = inp.shape[0]
                    
                    if not separate_var:
                        E_curr = np.zeros((len(pattern), batch_curr_size))
                        E_curr_l1 = np.zeros((len(pattern), batch_curr_size))
                    else:
                        E_curr = np.zeros((len_sep, len(pattern), batch_curr_size))
                        E_curr_l1 = np.zeros((len_sep, len(pattern), batch_curr_size))
                        
                # --- SCALAR TIME ARGUMENT ---
                # This should also represent the JUMP (dt), not absolute time.
                # 'zero_time_inp' is 0. 'jump*delta' is dt.
                time_batch = zero_time_inp + jump * delta 
                if steady: time_batch = zero_time_inp + 1.0

                out = copy.deepcopy(output_batch_[acc_jump - 1::time_steps]).to(device)
                
                # --- PREDICT ---
                pred = model(inp, time_batch)

                # --- ERROR CALCULATION ---
                if not separate_var:
                    err = (torch.mean(abs(pred - out), (-3, -2, -1)) / torch.mean(abs(out), (-3, -2, -1))) * 100
                    E_curr[i] = err.detach().cpu().numpy()
                    err_l1 = torch.mean(abs(pred - out), (-3, -2, -1))
                    E_curr_l1[i] = err_l1.detach().cpu().numpy()
                else:
                    for j in range(len_sep):
                        start_c = diff[j]
                        end_c = diff[j+1]
                        p_var = pred[:, start_c:end_c]
                        o_var = out[:, start_c:end_c]
                        err = (torch.mean(abs(p_var - o_var), (-3, -2, -1)) / (torch.mean(abs(o_var), (-3, -2, -1)) + 1e-10)) * 100
                        E_curr[j, i] = err.detach().cpu().numpy()
                        err_l1 = torch.mean(abs(p_var - o_var), (-3, -2, -1))
                        E_curr_l1[j, i] = err_l1.detach().cpu().numpy()
                        
                # --- UPDATE INPUT FOR NEXT STEP ---
                
                # 1. Update Wave 'u' (Channel 0) from Prediction
                inp[:, 0:1] = pred[:, 0:1] * (std_out/std_inp)
                
                # 2. Reset Velocity 'c' (Channel 1) from Static Truth
                if separate_var and len(separate_dim) > 1:
                    inp[:, 1:2] = static_c_channel
                else:
                    inp[:, 1:2] = pred[:, 1:2] 

                # 3. Reset Time 'dt' (Channel 2) from Static Truth
                # DO NOT ACCUMULATE TIME. Keep it as the step size.
                if inp.shape[1] > 2:
                    inp[:, 2:3] = static_dt_channel
                
            E = np.concatenate((E, E_curr), axis=-1)
            E_l1 = np.concatenate((E_l1, E_curr_l1), axis=-1)
            
    return T, E, E_l1

def _evaulate_models(folders, which_example="wave_gauss", cluster=False, in_dist=True,
                     patterns=[[1]], folder_pattern="errors", additional_flags=[],
                     steps=10, steady=False, separate_var=False, is_masked=False,
                     fine_tuned=False, training_folder=None, write_in_file=True,
                     old_in_dim=5, new_in_dim=5, new_out_dim=4):
    
    base_folder = "/cluster/home/svisnjic/Thalassa/results/checkpoints"
    ckpt_file = "epoch=45-step=100096.ckpt"
    model_dirs = [(base_folder, ckpt_file)]

    if in_dist: in_dist_sufix = "in_dist_patterns.txt"
    else: in_dist_sufix = "out_dist_patterns.txt"
    
    for folder, _model_file in model_dirs:
        print(f"Evaluating Model in: {folder}")
        
        if not fine_tuned and training_folder is None: _folder = folder
        else: _folder = training_folder
        
        train_file = os.path.join(_folder, "model_1401", "training_properties.txt")
        net_file   = os.path.join(_folder, "model_1401", "net_architecture.txt")
        ckpt_path  = os.path.join(folder, "model_1401", _model_file)
        
        print("Initializing WaveLoader...")
        loader = WaveLoader(train_file, net_file, ckpt_path, which_name=which_example)
        
        try: model = loader.load_model()
        except FileNotFoundError as e:
            print(e); continue
            
        loader_dict = loader.loader_dict
        device = loader.device
        
        raw_bs = 32
        if raw_bs % steps != 0:
            bs = (raw_bs // steps) * steps
            if bs == 0: bs = steps
        else: bs = raw_bs
        
        print("Loading Test Data...")
        test_data_loader = _load_data(which=which_example, loader_dict=loader_dict, batch_size=bs, which_loader="test", fix_input_to_time_step=0) 
        
        if separate_var: separate_dim = loader_dict["separate_dim"]
        else: separate_dim = []
        
        output_dir = os.path.join(folder, folder_pattern)
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        
        for pattern in patterns:
            print(f"\n--- Running Pattern: {pattern} ---")
            T, E, E_l1 = _test_pattern(model=model, testing_loader=test_data_loader, dt=loader_dict["dt"],
                                       time_steps=loader_dict["time_steps"], max_steps=loader_dict["step_max"],
                                       pattern=pattern, device=device, batch_size=bs, folder=output_dir,
                                       separate_var=separate_var, separate_dim=separate_dim,
                                       is_masked=is_masked, steady=steady, is_airfoil=(which_example == "airfoil"))
            
            s = "".join([str(p) + "_" for p in pattern])
            error_file = os.path.join(output_dir, s + in_dist_sufix)
            if write_in_file:
                _write_in_file(error_file, loader_dict, E, E_l1, T, pattern, separate_var=separate_var, num_var=len(separate_dim))
        
        print("\nEvaluation Complete.")
        print("----------------------------------")

if __name__ == "__main__":
    _evaulate_models(folders="/cluster/home/svisnjic/Thalassa/results/checkpoints",
                     which_example="wave_gauss", cluster=True, in_dist=True,
                     patterns=[[1] * 15, [3, 3, 3, 3, 3], [5, 5, 5], [15]],
                     folder_pattern="errors", steps=15, additional_flags=[],
                     steady=False, separate_var=True, is_masked=False, write_in_file=True)