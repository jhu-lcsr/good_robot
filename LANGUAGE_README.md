# Language code README

## Files
Code is organized into several (currently) disjoint files from the rest of the code

- `data.py`: handles ingesting dataset, reading trajectories, turning those trajectories into batches, shuffling, tensorizing, obtaining a vocab, etc. 
- `language_embedders.py`: contains classes for embedding, currently only implementing `RandomEmbedder` 
- `encoders.py`: contains encoder classes, currently only implementing `LSTMEncoder` 
- `language.py`: DEPRECATED but has some code stubs that may be useful later on so i havent removed it yet 
- `unet_module.py`: implements several variants of the UNet architecture, culminating in `UNetWithLanguage` which includes basic concatenative language fusion  from LSTM states 
- `unet_shared.py`: combines two unet modules for predicting previous and next states and handles sharing parameters between the two 
- `train_language_encoder.py`: Contains base trainer classes for training a `LanguageEncoder` (deprecated) but new trainers inherit from this, as it has code for generating images, saving checkpoints, loading checkpoints, etc. 
- `train_unet.py`: instatiates trainer class for a `SharedUNet` with modified loss computation that computes loss for previous and next prediction 

## Scripts 
Some useful scripts are located in `grid_scripts/*` that show how to run the code. Specifically, `grid_scripts/debug_unet.py` is a script to overfit to a tiny subset of the dev data, useful for making sure shapes line up, data is processed correctly, etc. 

## Data
All data is located in the `blocks_data/` dir. Organization is: 
- `blocks_data/*.json`: the split in JSON format (splits are `train`, `dev`, `test`, `single` (one single trajectory), and `tiny` (small subset of dev trajectories). These are the files that are injested by `data.py`
- `blocks_data/blocks_objs` contains the block .obj files with their corresponding mesh files 
- `blocks_data/color_logos` has a couple useful scripts for writing mesh files automatically 


## TODOs 
- implement IoU metric for evaluating output quality
- try one-hot encoding of input (done) 
- lower capacity of language embedder/encoder 
