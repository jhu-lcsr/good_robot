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

python -u train_language_encoder.py \
        --train-path blocks_data/tinyset.json \
        --val-path blocks_data/tinyset.json \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --num-epochs 202 \
        --num-blocks 20 \
        --generate-after-n 200\
        --traj-type flat \
        --batch-size 256  \
        --max-seq-length 40 \
        --embedding-dim 16 \
        --encoder-hidden-dim 16 \
        --encoder-num-layers 2 \
        --deconv decoupled \
        --deconv-factor 1 \
        --fuser tiled \
        --mlp-hidden-dim 32 \
        --mlp-num-layers 2 \
        --mlp-dropout 0.0 \
        --dropout 0.0 \
        --bidirectional \
        --cuda 0  | tee ${CHECKPOINT_DIR}/stdout.log
        #--compute-block-dist \
