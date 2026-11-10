import os
import sys
import h5py
import torch
import numpy as np

from inv_wave_loader import InverseWaveGaussians

def _load_inverse_dataset(dic, 
                  which, 
                  which_loader,
                  in_dim,
                  out_dim,
                  masked_input = None,
                  fix_input_to_time_step = None):
    
    num_samples = dic["num_samples"]

    
    print("WHICH: ", which, " NUM_SAMPLES: ", num_samples)

    if which == "wave_gauss":
        train_dataset =   InverseWaveGaussians(max_num_time_steps = dic["time_steps"], 
                                        time_step_size = dic["dt"],
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_loader,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = num_samples,
                                        data_path = "/cluster/home/svisnjic/Thalassa/data", # "---- PROVIDE THE FOLDER PATH ----",
                                        time_input = dic["time_input"],
                                        masked_input = masked_input,
                                        allowed_transitions = dic["allowed_tran"])

    
    # elif which == "wave_seismic": # need to do for seismic too...
    #     train_dataset =   WaveSeismic(max_num_time_steps = dic["time_steps"], 
    #                                 time_step_size = dic["dt"],
    #                                 fix_input_to_time_step = fix_input_to_time_step,
    #                                 which = which_loader,
    #                                 resolution = 128,
    #                                 in_dist = True,
    #                                 num_trajectories = num_samples,
    #                                 data_path = "/cluster/home/svisnjic/Thalassa/data", # "---- PROVIDE THE FOLDER PATH ----",
    #                                 time_input = dic["time_input"],
    #                                 masked_input = masked_input,
    #                                 allowed_transitions = dic["allowed_tran"])

    else:
        raise ValueError("Not implemented experiment")

    return train_dataset

def _load_inverse_dict(files, which_example, steps=15, is_masked=None):
    d = dict()

    d["which"] = which_example

    for file in files:
        with open(file) as f:
            lines = f.readlines()
        f.close()
        
        for line in lines:
            l = line.strip().split(",")
            if l[0] in ["epochs", "batch_size", "exp", "training_samples", "time_steps", "dt"]:
                value = int(float(l[1]))
            elif l[0] == "nl_dim":
                s = ""
                for a in l[1:]:
                    if a[0]=="\"":
                        a = a.strip(" ")
                        s+=a[-1]
                    elif a[-1] == "\"":
                        a = a.strip(" ")
                        s+=a[0]
                    else:
                        s+=a.strip(" ")
                value = s
                S = value.split(", ")
                string = ""
                for s in S:
                    string+=s
                value = string
                
            elif "." in l[1]:
                value = float(l[1])
            elif "False" in l[1]:
                value = False
            elif "True" in l[1]:
                value = True
            elif "relu" in l[1]:
                value = l[1]
            elif "e-" in l[1]:
                value = float(l[1])
            elif "allowed" in l[0]:
                value = l[1]
            else:
                value = int(l[1])
            d[l[0]] = value

    #------------------
    
    if "batch_norm" in d:
        d["batch_norm"] = True if d["batch_norm"] in [1, True] else False
    if "time_input" in d:
        d["time_input"] = True if d["time_input"] in [1, True] else False
    if "cluster" in d:
        d["cluster"]    = True if d["cluster"]    in [1, True] else False
    
    #------------------

    if "nl_dim" in d:
        if d["nl_dim"] in ["023"]:
            d["nl_dim"] = [0,2,3]
        elif d["nl_dim"] in ["123"]:
            d["nl_dim"] = [1,2,3]
        elif d["nl_dim"] in ["23"]:
            d["nl_dim"] = [2,3]
    
    #------------------
    
    if "is_att" in d:
        d["is_att"] = True if d["is_att"] in [1, True] else False
    else:
        d["is_att"] = False
        d["patch_size"] = None
        d["dim_multiplier"] = None
        d["depth"] = None
        d["heads"] = None
        d["dim_head_multiplier"] = None
        d["mlp_dim_multiplier"] = None
        d["emb_dropout"] = None
    
    #------------------
    
    d["masked_input"] = None
    if is_masked:
        d["is_masked"] = is_masked 
    else:
        d["is_masked"] = None 

    if which_example in ['wave_gauss']:
        d['in_dim'] = 3 # u(t1), u(t2), dt= t2-t1  are the inputs # if d['time_input'] == True else 2 # this shouldn't happen (copy paste redundancy)
        d['out_dim'] = 1 # the output dim is one -> predict only c
        d['separate'] = False
        d["separate_dim"] = [1] # there is no separate physical quantity, only c

        d['nmax'] = 10512
        d['step_max'] = 15 # specific for Gauss
    
    else:
        raise ValueError("Please specify different benchmark")

    assert "in_dim" in d
    assert "out_dim" in d
    
    d['include_zero'] = False

    d["time_steps"] = steps
    _allowed = []
    print("STEPS", d["time_steps"])
    d["allowed"] = 'all'
    if d['allowed'] == 'all':
        for t in range(1,d["time_steps"]+1):
            _allowed.append(t)
    elif d['allowed'] == 'one':
        _allowed = [1]
    d["allowed_transitions"] = _allowed
    print(_allowed)
    if "exp" not in d:
        d["exp"] = 1

    return d
