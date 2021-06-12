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

#CHECKPOINT_DIR="models/transformer_models/oh_patch_dropout_${DROPOUT}_weighted_${WEIGHT}_lr_${LR}"

mkdir -p ${CHECKPOINT_DIR}/code
# copy all code 
cp *.py ${CHECKPOINT_DIR}/code/
# record git info 
echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

#python make_config.py ${CONFIG} ${OUTCONFIG} ${CHECKPOINT_DIR}

python -u train_transformer.py \
        --cfg ${CONFIG} \
        --resume \
        --checkpoint-dir ${CHECKPOINT_DIR} | tee ${CHECKPOINT_DIR}/stdout.log
