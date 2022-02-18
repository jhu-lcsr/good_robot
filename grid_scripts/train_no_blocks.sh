#!/bin/bash
#$ -j yes
#$ -N train_blocks
#$ -l 'mem_free=80G,h_rt=13:00:00,gpu=1'
#$ -q gpu.q@@rtx 
#$ -m ae -M elias@jhu.edu
#$ -cwd
#$ -o /home/hltcoe/estengel/real_good_robot/grid_logs/train.out


ml cuda10.0/toolkit
ml cudnn/7.5.0_cuda10.0

source activate blocks 


mkdir -p ${CHECKPOINT_DIR}/code
# copy all code 
cp *.py ${CHECKPOINT_DIR}/code/
# record git info 
echo "RUNNING ON VERSION: " > ${CHECKPOINT_DIR}/stdout.log
git branch >> ${CHECKPOINT_DIR}/stdout.log
git reflog | head -n 1 >> ${CHECKPOINT_DIR}/stdout.log

python -u train_language_encoder.py \
        --train-path blocks_data/small_trainset.json \
        --val-path blocks_data/devset.json \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --num-epochs 100  \
        --num-blocks 20 \
        --generate-after-n 200 \
        --traj-type flat \
        --bidirectional \
        --batch-size 128 \
        --max-seq-length 40 \
        --deconv decoupled \
        --deconv-factor 1 \
        --fuser tiled \
        --cuda 0 | tee ${CHECKPOINT_DIR}/stdout.log 

