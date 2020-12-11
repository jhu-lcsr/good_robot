import sys
import pathlib 
import yaml 

if __name__ == "__main__":
    config_path = sys.argv[1]
    out_config_path = sys.argv[2]
    ckpt_dir = sys.argv[3]
    with open(config_path) as f1:
        config = yaml.load(f1, Loader=yaml.FullLoader) 

    config["checkpoint_dir"] = ckpt_dir 
    
    #new_config_path = pathlib.Path(config_path)
    #config_name = new_config_path.name 
    #config_home = new_config_path.parent
    #config_name = f"edited_{config_name}"
    #new_config_out = config_home.joinpath(config_name) 

    with open(out_config_path, "w") as f1:
        yaml.dump(config, f1) 
