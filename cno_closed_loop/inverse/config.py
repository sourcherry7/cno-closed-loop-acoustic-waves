# Experiment configuration and dataset factory for the inverse problem.
#
# Adapted from camlab-ethz/ConvolutionalNeuralOperator (MIT License), commit 75a3b9b:
#   _load_inverse_dict    follows test_and_fine_tune_utils/test_utils.py::_load_dict
#   _load_inverse_dataset follows DataLoaders/load_utils.py::_load_dataset
# Changes: inverse-problem I/O (inputs u(t1), u(t2), dt; output c) for the Wave-Gauss data.

from cno_closed_loop import paths
from cno_closed_loop.inverse.dataset import InverseWaveGaussians


def _load_inverse_dataset(dic, 
                  which, 
                  which_loader,
                  in_dim,
                  out_dim,
                  masked_input = None,
                  fix_input_to_time_step = None):
    """Inverse-problem dataset for the given split ("train", "val" or "test")."""
    num_samples = dic["num_samples"]
    print(f"Inverse dataset '{which}' ({which_loader}): {num_samples} trajectories")

    if which == "wave_gauss":
        train_dataset =   InverseWaveGaussians(max_num_time_steps = dic["time_steps"], 
                                        time_step_size = dic["dt"],
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_loader,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = num_samples,
                                        data_path = str(paths.DATA_DIR),
                                        time_input = dic["time_input"],
                                        masked_input = masked_input,
                                        allowed_transitions = dic["allowed_tran"])
    else:
        raise ValueError("Not implemented experiment")

    return train_dataset


def _load_inverse_dict(files, which_example, steps=15, is_masked=None):
    """Experiment configuration for the inverse problem, from a run's property files."""
    d = dict()

    d["which"] = which_example

    for file in files:
        with open(file) as f:
            lines = f.readlines()
        
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
        d['in_dim'] = 3 # inputs: u(t1), u(t2) and dt = t2 - t1
        d['out_dim'] = 1 # output: c
        d['separate'] = False
        d["separate_dim"] = [1] # a single output quantity

        d['nmax'] = 10512
        d['step_max'] = 15 # time steps of the Wave-Gauss trajectories
    
    else:
        raise ValueError("Please specify different benchmark")

    assert "in_dim" in d
    assert "out_dim" in d
    
    d['include_zero'] = False

    d["time_steps"] = steps
    _allowed = []
    d["allowed"] = 'all'
    if d['allowed'] == 'all':
        for t in range(1,d["time_steps"]+1):
            _allowed.append(t)
    elif d['allowed'] == 'one':
        _allowed = [1]
    d["allowed_transitions"] = _allowed
    if "exp" not in d:
        d["exp"] = 1

    return d
