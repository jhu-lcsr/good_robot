#!/bin/bash

# script to copy over just what is needed, not unnecessary checkpoints 
DIR=$1
DEST=$2

alias rclone="/home/estengel/sbin/rclone"

/home/estengel/sbin/rclone mkdir remote:${DEST}
/home/estengel/sbin/rclone  copy ${DIR}/code remote:${DEST}/code
/home/estengel/sbin/rclone copy ${DIR}/stdout.log remote:${DEST}
/home/estengel/sbin/rclone copy ${DIR}/vocab.json remote:${DEST}
/home/estengel/sbin/rclone copy ${DIR}/best_training_state.json remote:${DEST}
/home/estengel/sbin/rclone copy ${DIR}/config.yaml remote:${DEST}
/home/estengel/sbin/rclone copy ${DIR}/best.th remote:${DEST}
