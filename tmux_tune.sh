#!/bin/bash

for num in $(seq 0 7);
do 
    #tmux new-session -d -s "test${num}" "bash execute_tune.sh ${num}";
    tmux new-session -d -s "test${num}" "bash execute_tune_gr.sh ${num}";
    sleep 60;
done;

