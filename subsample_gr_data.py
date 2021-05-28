import pathlib
import numpy as np
import pickle as pkl

original = pathlib.Path("/srv/local1/estengel/gr_data")
out_dir = pathlib.Path("/srv/local1/estengel/gr_subsets") 
for perc in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    for pkl_name in original.glob("*/*.pkl"):
        with open(pkl_name, 'rb') as f1:
            data = pkl.load(f1)
        data_len = len(data) 
        perc_float = perc/100
        subset_n = int(data_len * perc_float) 
        subset_idxs = np.random.choice(data_len, size = subset_n, replace=False) 
        subset_data = [data[idx] for idx in subset_idxs]
        out_name  = out_dir.joinpath(f"{perc}") 
        
        parent_name = pkl_name.parent.name 
        out_name = out_name.joinpath(parent_name).joinpath("with_actions.pkl") 
        out_name.parent.mkdir(parents=True) 
        with open(out_name, "wb") as f1:
            pkl.dump(subset_data, f1) 
    
