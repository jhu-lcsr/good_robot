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

CHECKPOINT_DIR="models/language_pretrain"

mkdir -p ${CHECKPOINT_DIR}/code
# copy all code 
cp *.py ${CHECKPOINT_DIR}/code/
# record git info 
echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

python -u train_transformer.py \
        --train-path blocks_data/singleset.json \
        --val-path blocks_data/singleset.json \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --num-epochs 210 \
        --num-blocks 1 \
        --binarize-blocks \
        --generate-after-n 208 \
        --traj-type flat \
        --batch-size 256  \
        --max-seq-length 40 \
        --do-filter \
        --top-only \
        --embedding-dim 16 \
        --embedder random \
        --embedding-dim 50 \
        --hidden-dim 64 \
        --n-heads 4 \
        --n-layers 3 \
        --ff-dim 256 \
        --learn-rate 0.001 \
        --dropout 0.0 \
        --embed-dropout 0.0 \
        --zero-weight 0.001 
