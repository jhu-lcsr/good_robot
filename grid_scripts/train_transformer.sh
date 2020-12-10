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

CHECKPOINT_DIR="models/transformer_models/dropout_${DROPOUT}_weighted_${WEIGHT}_lr_${LR}"

mkdir -p ${CHECKPOINT_DIR}/code
# copy all code 
cp *.py ${CHECKPOINT_DIR}/code/
# record git info 
echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

python -u train_transformer.py \
        --train-path blocks_data/trainset_v2.json \
        --val-path blocks_data/small_devset.json \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --num-epochs 400 \
        --num-blocks 1 \
        --binarize-blocks \
        --generate-after-n 408 \
        --traj-type flat \
        --batch-size 768 \
        --max-seq-length 60 \
        --do-filter \
        --top-only \
        --embedder glove \
        --embedding-dim 100 \
        --embedding-file /home/estengel/glove/glove.6B.100d.txt \
        --hidden-dim 256 \
        --n-heads 4 \
        --n-layers 6 \
        --ff-dim 1024 \
        --learn-rate ${LR} \
        --dropout ${DROPOUT} \
        --embed-dropout ${DROPOUT} \
        --zero-weight ${WEIGHT} \
        --cuda 0 | tee ${CHECKPOINT_DIR}/stdout.log
