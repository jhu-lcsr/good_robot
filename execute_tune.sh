#!/bin/bash

NUM=$1

all_files=$(ls tune_files/${NUM}/*.txt)


for file in ${all_files}; 
do

    setting_str=$(cat ${file}) 
    stringarray=($setting_str)
    dr=${stringarray[0]}
    n_sh=${stringarray[1]}
    n_sp=${stringarray[2]}
    nh=${stringarray[3]}
    warm=${stringarray[4]}
    w=${stringarray[5]}
    
    #export CONFIG=configs/patch_transformer/tune_fixed.yaml;
    #export CONFIG=configs/image_transformer/bert.yaml
    #export CONFIG=configs/image_transformer/tune32mm_glove_no_recon.yaml
    #export CONFIG=configs/image_transformer/tune32mm_bert_no_recon.yaml
    export CONFIG=configs/image_tuning_bisk/bert_no_recon_for_tuning.yaml
    dir="/srv/local1/estengel/models/transformer_tune_images_bert_no_recon/${dr}_${w}_${n_sh}_${n_sp}_${nh}_${warm}";
    export CHECKPOINT_DIR=${dir};
    export ZERO_WEIGHT=${w};
    export SHARED_LAYERS=${n_sh};
    export SPLIT_LAYERS=${n_sp};
    export NHEADS=${nh};
    export WARMUP=${warm} 
    echo ${CHECKPOINT_DIR} &> grid_logs/tmux_test_${NUM} 
    echo "executing grid_scripts in $(pwd)" >> grid_logs/tmux_test_${NUM};
    bash grid_scripts/tune_transformer.sh;
    #echo "cuda ${CUDA_VISIBLE_DEVICE}" >> grid_logs/tmux_test_${NUM}
done;

