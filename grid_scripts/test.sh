#!/bin/bash
#$ -j yes
#$ -N train_blocks
#$ -l 'mem_free=80G,h_rt=13:00:00,gpu=1'
#$ -q gpu.q@@rtx 
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/real_good_robot/grid_logs/test.out 


ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

source activate blocks 

#CHECKPOINT_DIR="/exp/estengel/blocks_project/models/lstm_vocab"

# record git info 
echo "RUNNING ON VERSION: " >> ${CHECKPOINT_DIR}/testout.log
git branch >> ${CHECKPOINT_DIR}/testout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/testout.log

echo "DECODING"
python -u train_language_encoder.py \
        --train-path blocks_data/trainset_v2.json  \
        --val-path blocks_data/tinyset.json \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --num-epochs 100  \
        --num-blocks 20 \
        --generate-after-n 1 \
        --traj-type flat \
        --bidirectional \
        --batch-size 128 \
        --max-seq-length 40 \
        --deconv decoupled \
        --deconv-factor 1 \
        --fuser tiled \
        --test \
        --cuda 0 


