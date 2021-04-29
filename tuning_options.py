import random 
import sys
import json
from itertools import product
import numpy as np
import pdb 
import pathlib 
np.random.seed(12) 

num_workers = 7
jobs_per_worker = 10

total_jobs = num_workers * jobs_per_worker 

dropouts = [ 0.20, 0.33, 0.40]
n_shared_layers = [1, 2, 4, 6]
n_split_layers = [1, 2, 4, 6] 
n_heads = [2, 4, 8]
warmups = [100, 400, 800, 1000, 2000, 4000] 
zero_weight = [0.1]
init_scale = [4, 16, 64]

all_combos = product(dropouts, n_shared_layers, n_split_layers, n_heads, warmups, zero_weight, init_scale) 
all_combos = [x for x in all_combos]
all_ids = [i for i in range(len(all_combos))]
chosen_combos = [all_combos[i] for i in np.random.choice(all_ids, total_jobs, replace = False)]

for i in range(num_workers): 
    start = i * jobs_per_worker
    end = (i+1) * jobs_per_worker
    settings = chosen_combos[start:end]
   
    settings_dir = pathlib.Path(f"tune_files/{i}")
    settings_dir.mkdir(exist_ok=True) 
    for j, setting in enumerate(settings): 
        setting = [str(x) for x in setting]
        with open(settings_dir.joinpath(f"{j}.txt"), "w") as f1:
            f1.write(" ".join(setting))







