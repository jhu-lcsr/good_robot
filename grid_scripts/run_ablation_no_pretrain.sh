#!/bin/bash

PARENT_DIR=/srv/local1/estengel/models/gr_real_data/stacks_ablation_no_pretrain
CONFIG_DIR=configs/gr_transformer/stacks_real_data/ablation_no_pretrain

for perc in 10 20 30 40 50 60 70 80 90 100 
do 
    export CHECKPOINT_DIR=${PARENT_DIR}/${perc}
    export CONFIG=${CONFIG_DIR}/${perc}.yaml

    echo ${CHECKPOINT_DIR}
    echo ${CONFIG}
    ./grid_scripts/train_gr.sh 
    ./grid_scripts/test_gr.sh 
done
