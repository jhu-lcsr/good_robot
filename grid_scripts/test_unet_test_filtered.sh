#!/bin/bash
#$ -j yes
#$ -N train_blocks
#$ -l 'mem_free=80G,h_rt=13:00:00,gpu=1'
#$ -q gpu.q@@rtx 
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/real_good_robot/grid_logs/train.out


#ml cuda10.0/toolkit
#ml cudnn/7.5.0_cuda10.0

source activate blocks 

python -u train_unet.py  \
    --cfg ${CHECKPOINT_DIR}/config.yaml \
    --test \
    --test-path blocks_data/filtered/test.jsonlines \
    --out-path ${CHECKPOINT_DIR}/filtered_test_metrics.json \
    --batch-size 1 \
    --generate-after-n 1000
