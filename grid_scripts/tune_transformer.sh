#!/bin/bash

source activate blocks 

echo "MAKING DIR ${CHECKPOINT_DIR}" >> grid_logs/log.out 

mkdir -p ${CHECKPOINT_DIR}/code
## copy all code 
cp *.py ${CHECKPOINT_DIR}/code/
# record git info 
echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

python -u train_transformer.py \
        --cfg ${CONFIG} \
        --n-shared-layers ${SHARED_LAYERS} \
        --n-split-layers ${SPLIT_LAYERS} \
        --n-heads ${NHEADS} \
        --warmup ${WARMUP} \
        --zero-weight ${ZERO_WEIGHT} \
        --checkpoint-dir ${CHECKPOINT_DIR} | tee ${CHECKPOINT_DIR}/stdout.log 
        #--num-epochs 15 \
        #--batch-size 64 \
