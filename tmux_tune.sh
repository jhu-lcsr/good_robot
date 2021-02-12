#!/bin/bash

for num in $(seq 0 4);
do 
    tmux new-session -d -s "test${num}" "bash execute_tune.sh ${num}";
done;

