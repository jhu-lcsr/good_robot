#!/bin/sh

#    Suggested commands:
#        export CUDA_VISIBLE_DEVICES="0" && ./finetune.sh -f -e -s -t row
#        export CUDA_VISIBLE_DEVICES="0" && ./finetune.sh -f -e -s -t stack
#        export CUDA_VISIBLE_DEVICES="0" && ./finetune.sh -f -e -s -t unstack
#        export CUDA_VISIBLE_DEVICES="0" && ./finetune.sh -f -e -s -t vertical_square
# don't finetune by default (assume it hasn't been done)
finetune=0;
embed=0;
ssr=0;
task="";

# make sure script will print where it is and exit on errors
set -e
set -u
set -x

# help function
Help()
{
   # Display Help
   echo "Syntax: ./finetune.sh [-f|t|e|s|h]"
   echo ""
   echo "options:"
   echo "-f     Finetune policies from base_models. Other options will break if this hasn't been run at least once."
   echo "-t     Which task? [row|stack|unstack|vertical_square]"
   echo "-e     Precompute demo embeddings. If you do not already have embed.pickle files in the demo folders, this is highly recommended."
   echo "-s     Run SSR Simulation Experiments."
   echo "-h     Print help menu."
   echo
   echo "Suggested commands:"
   echo "    export CUDA_VISIBLE_DEVICES=\"0\" && ./finetune.sh -f -e -s -t row"
   echo "    export CUDA_VISIBLE_DEVICES=\"0\" && ./finetune.sh -f -e -s -t stack"
   echo "    export CUDA_VISIBLE_DEVICES=\"0\" && ./finetune.sh -f -e -s -t unstack"
   echo "    export CUDA_VISIBLE_DEVICES=\"0\" && ./finetune.sh -f -e -s -t vertical_square"
   echo
   echo "Note that running without any of the options set will result in no action."
}

# get cmd line options
while getopts "t:fesh" flag
do
    case $flag in
        f) # run finetuning?
            finetune=1;;
        t) # which task?
            task=${OPTARG};;
        e) # compute embeddings?
            embed=1;;
        s) # run ssr?
            ssr=1;;
        h) # help
            Help
            exit;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

if [ $finetune -eq 1 ]
then
    echo "Running Finetuning..."
    if [ "$task" = "row" ]
    then
        # row finetunes
        python3 train_offline.py -m logs/base_models/stacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/row_demos/ -t row -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/unstacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/row_demos/ -t row -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/vertical_square_hist_densenet/snapshot.reinforcement_trial_success_rate_best_value.pth -d demos/row_demos/ -t row -o logs/finetuned_models/
    elif [ "$task" = "stack" ]
    then
        # stack finetunes
        python3 train_offline.py -m logs/base_models/rows_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/stack_demos/ -t stack -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/unstacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/stack_demos/ -t stack -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/vertical_square_hist_densenet/snapshot.reinforcement_trial_success_rate_best_value.pth -d demos/stack_demos/ -t stack -o logs/finetuned_models/

    elif [ "$task" = "unstack" ]
    then
        # unstack finetunes
        python3 train_offline.py -m logs/base_models/rows_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/unstacking_demos/ -t unstack -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/stacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/unstacking_demos/ -t unstack -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/vertical_square_hist_densenet/snapshot.reinforcement_trial_success_rate_best_value.pth -d demos/unstacking_demos/ -t unstack -o logs/finetuned_models/

    elif [ "$task" = "vertical_square" ]
    then
        # vertical square finetunes
        python3 train_offline.py -m logs/base_models/rows_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/vertical_square_demos/ -t vertical_square -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/stacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/vertical_square_demos/ -t vertical_square -o logs/finetuned_models/
        python3 train_offline.py -m logs/base_models/unstacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/vertical_square_demos/ -t vertical_square -o logs/finetuned_models/

    else
        echo "Must pass one of [row | stack | unstack | vertical_square] to -t."
    fi
fi

# regenerate embedding pickles

if [ $embed -eq 1 ]
then
    echo "Computing and Saving Embeddings..."
    if [ "$task" = "row" ]
    then
        # row
        python3 evaluate_demo_correspondence.py -e demos/row_demos -d demos/row_demos -t row --stack_snapshot_file logs/finetuned_models/base_stack_finetune_row.pth \
        --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_row.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_row.pth \
        --write_embed --depth_channels_history --cycle_consistency
    elif [ "$task" = "stack" ]
    then
        # stack
        python3 evaluate_demo_correspondence.py -e demos/stack_demos -d demos/stack_demos -t stack --row_snapshot_file logs/finetuned_models/base_row_finetune_stack.pth \
        --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_stack.pth \
        --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_stack.pth --write_embed --depth_channels_history --cycle_consistency

    elif [ "$task" = "unstack" ]
    then
        # unstack
        python3 evaluate_demo_correspondence.py -e demos/unstacking_demos -d demos/unstacking_demos -t unstack --stack_snapshot_file \
        logs/finetuned_models/base_stack_finetune_unstack.pth --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_unstack.pth \
        --row_snapshot_file logs/finetuned_models/base_row_finetune_unstack.pth \
        --write_embed --depth_channels_history --cycle_consistency

    elif [ "$task" = "vertical_square" ]
    then
        # vertical_square
        python3 evaluate_demo_correspondence.py -e demos/vertical_square_demos -d demos/vertical_square_demos -t vertical_square --row_snapshot_file \
        logs/finetuned_models/base_row_finetune_vertical_square.pth --stack_snapshot_file logs/finetuned_models/base_stack_finetune_vertical_square.pth \
        --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_vertical_square.pth --write_embed --depth_channels_history --cycle_consistency
    else
        echo "Must pass one of [row | stack | unstack | vertical_square] to -t."
    fi
fi

# Run SSR
if [ $ssr -eq 1 ]
then
    ./ssr.sh -t $task
fi
