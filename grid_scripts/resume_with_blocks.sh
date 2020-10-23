#!/bin/bash
#$ -j yes
#$ -N train_blocks
#$ -l 'mem_free=80G,h_rt=24:00:00,gpu=1'
#$ -q gpu.q@@rtx 
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/real_good_robot/grid_logs/train.out


ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

source activate blocks 

python -u train_language_encoder.py \
        --train-path blocks_data/trainset_v2.json  \
        --val-path blocks_data/devset.json \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --num-epochs 100  \
        --num-blocks 20 \
        --generate-after-n 200 \
        --traj-type flat \
        --bidirectional \
        --batch-size 256 \
        --max-seq-length 40 \
        --compute-block-dist \
        --resume \
        --cuda 0 &> ${CHECKPOINT_DIR}/stdout.log 
