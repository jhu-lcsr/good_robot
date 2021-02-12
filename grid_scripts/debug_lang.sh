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

CHECKPOINT_DIR="models/debug_lang"

mkdir -p ${CHECKPOINT_DIR}/code
# copy all code 
cp *.py ${CHECKPOINT_DIR}/code/
# record git info 
echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

python -u train_language_only.py \
        --train-path blocks_data/singleset.json \
        --val-path blocks_data/singleset.json \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --num-epochs 110 \
        --num-blocks 20 \
        --binarize-blocks \
        --traj-type flat \
        --batch-size 16 \
        --max-seq-length 40 \
        --do-filter \
        --top-only \
        --embedding-dim 16 \
        --encoder-hidden-dim 16 \
        --encoder-num-layers 2 \
        --mlp-hidden-dim 32 \
        --mlp-num-layers 2 \
        --dropout 0.0 \
        --bidirectional \
        --cuda 0  
