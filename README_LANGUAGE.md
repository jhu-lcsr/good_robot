# Language code README

## Files
Code is organized into several (currently) disjoint files from the rest of the code

- `data.py`: handles ingesting dataset, reading trajectories, turning those trajectories into batches, shuffling, tensorizing, obtaining a vocab, etc. 
- `language_embedders.py`: contains classes for embedding, currently only implementing `RandomEmbedder` 
- `unet_module.py`: implements several variants of the UNet architecture, culminating in `UNetWithLanguage` which includes basic concatenative language fusion  from LSTM states 
- `unet_shared.py`: combines two unet modules for predicting previous and next states and handles sharing parameters between the two 
- `train_language_encoder.py`: Contains base trainer classes for training a `LanguageEncoder` (deprecated) but new trainers inherit from this, as it has code for generating images, saving checkpoints, loading checkpoints, etc. 
- `train_unet.py`: instatiates trainer class for a `SharedUNet` with modified loss computation that computes loss for previous and next prediction 
- `transformer.py`: code for building a transformer model, tiling an image into patches, and re-combining patches back into an image
- `train_transformer.py`: trainer class for a transformer model, with command-line interface; takes care of checkpointing, loading, etc.
- `metrics.py`: implements useful metrics, especially the TeleportationMetric, which 1) selects the block to move by taking the maximum `previous` pixel after intersection with true block locations, 2) selects the location to put it by taking the max region from the `next` pixelmap 3) executes the move by modifying the `previous` state representation to remove the block at the previous position and place it at the predicted next position and 4) evaluates the Euclidean distance between the center of the predicted block location and the center of the true block location, for the true block. Note that if the wrong block is selected to move, the distance will often be quite large since the true block has not moved. 


## Training a model 
To train a model, you can specify parameters using commandline parameters or a yaml config file. Use of a config file is recommended, as it is more replicable. The config format is `field: value` where `field` is one of the commandline arguments. 
Models are trained from the `train_<model>.py` files, i.e. the `main()` function of `train_transformer.py` is used to train a Transformer-based model. 
The path to the config can be passed to the `--cfg` parameter. Note that with the config files, all yaml arguments can be overwritten by using the command line. For example, `grid_scripts/train_transformer.sh` calls training with the following command: 

```
python -u train_transformer.py --cfg ${CONFIG} --checkpoint-dir ${CHECKPOINT_DIR}
```

This is desirable, since we might decide to train multiple models from the same config (for example, before/after a code change), but want separate output directories to store the checkpoints and logs. This lets us avoid rewriting the config file each time we want to change the output directory.
In general, the `checkpoint-dir` is where the `train_transformer.py` will store all logs, model checkpoints, and outputs, as well as a copy of the current version of the code. 
The best-performing checkpoint (chosen based on `--score-type`) will be saved in `${CHECKPOINT_DIR}/best.th` 

`train_transformer.py` has two other parameters that are useful to overwrite: `--resume` and `--test`. `--resume` resumes training an existing model. If this flag is not set and the `--checkpoint-dir` is non-empty, the program will error-out to avoid overwriting an existing model. If it is set, the script will load the parameters stored in `best.th` and training will resume from the epoch stored in `best_training_state.json`. 

### Training on GoodRobot Data
To train on the GoodRobot data, a different trainer is used (we need a different dataset reader, different metrics, etc. so this was easiest). 
You can run the same way as before, but now with 

```
python -u train_trainsformer_gr_data.py --cfg ${CONFIG} --checkpoint-dir ${CHECKPOINT_DIR}
```

The same command-line arguments apply as before, but the config files will look slightly different. Specifically, the `--path` argument should point to a parent dir where you have stored the GoodRobot runs, each run in a separate dir. 
These dirs need to be preprocessed using the notebook `blocks_data/explore_gr_data.ipynb` 
Another important flag is `--task-type` which tells the dataset reader 


## Evaluating a model 
If `--test` is set, the model will load from `best.th` and run all metrics. By default (if `--test-path` is set to None) it will evaluate against development data, saving all metrics to `val_metrics.json`.
If `--test-path` is set, the model will evaluate on test data and store to `test_metrics.json`. Note that this should ideally only be done once per model, as late as possible. 

## Generating images
Setting `--generate-after-n 0` and `--test`  will have the model `prev.png` and `next.png` images, that show the image-like state information, overlaid with the model prediction for a previous and next block location, respectively. 
These are color-coded based on the strength of the model prediction, dark red being a high value, white being 0. 
`next.png` also has a dark black X, which represents the center of the gold block location (i.e. where the block should go) and a large black diamond, which is the center of the predicted location of the correct block to move, computed via the teleportation process described in the `metrics.py` description. This info can help provide an intuitive guide for how good a particular prediction was, since you can see the distance between the predicted and gold location. 

## Scripts 
Some useful scripts are located in `grid_scripts/*` that show how to run the code. Specifically, `grid_scripts/debug_transformer.py` is a script to overfit to a tiny subset of the dev data, useful for making sure shapes line up, data is processed correctly, etc. 

## Data
All data is located in the `blocks_data/` dir. Organization is: 
- `blocks_data/*.json`: the split in JSON format (splits are `train`, `dev`, `test`, `single` (one single trajectory), and `tiny` (small subset of dev trajectories). These are the files that are injested by `data.py`
- `blocks_data/blocks_objs` contains the block .obj files with their corresponding mesh files 
- `blocks_data/color_logos` has a couple useful scripts for writing mesh files automatically 

