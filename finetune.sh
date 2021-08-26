#!/bin/sh

# NOTE This is what to modify if your paths are different.
PATH_TO_COPPELIA_SIM="~/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04/coppeliaSim.sh";
PATH_TO_RGR="~/real_good_robot";

# don't finetune by default (assume it hasn't been done)
finetune=0;
embed=0;
ssr=0;

# help function
Help()
{
   # Display Help
   echo "Syntax: ./finetune.sh [-f|e|t|h]"
   echo "options:"
   echo "-f     Finetune policies from base_models. Other options will break if this hasn't been run at least once."
   echo "-e     Precompute demo embeddings. If you do not already have embed.pickle files in the demo folders, this is highly recommended."
   echo "-t     Run SSR Simulation Experiments."
   echo "-h     Print help menu."
   echo
   echo "Note that running without any of the options set will result in no action."
}

# get cmd line options
while getopts ":hf:" option; do
    case $option in
        f) # run finetuning?
            finetune=1;;
        e) # compute embeddings?
            embed=1;;
        t) # run ssr?
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
    # row finetunes
    python train_offline.py -m logs/base_models/stacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/row_demos/ -t row -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/unstacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/row_demos/ -t row -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/vertical_square_hist_densenet/snapshot.reinforcement_trial_success_rate_best_value.pth -d demos/row_demos/ -t row -o logs/finetuned_models/

    # stack finetunes
    python train_offline.py -m logs/base_models/rows_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/stack_demos/ -t stack -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/unstacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/stack_demos/ -t stack -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/vertical_square_hist_densenet/snapshot.reinforcement_trial_success_rate_best_value.pth -d demos/stack_demos/ -t stack -o logs/finetuned_models/

    # unstack finetunes
    python train_offline.py -m logs/base_models/rows_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/unstacking_demos/ -t unstack -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/stacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/unstacking_demos/ -t unstack -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/vertical_square_hist_densenet/snapshot.reinforcement_trial_success_rate_best_value.pth -d demos/unstacking_demos/ -t unstack -o logs/finetuned_models/

    # vertical square finetunes
    python train_offline.py -m logs/base_models/rows_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/vertical_square_demos/ -t vertical_square -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/stacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/vertical_square_demos/ -t vertical_square -o logs/finetuned_models/
    python train_offline.py -m logs/base_models/unstacking_hist_densenet/snapshot.reinforcement_action_efficiency_best_value.pth -d demos/vertical_square_demos/ -t vertical_square -o logs/finetuned_models/

fi

# regenerate embedding pickles

if [ $embed -eq 1 ]
then
    echo "Computing and Saving Embeddings..."
    # row
    python3 evaluate_demo_correspondence.py -e demos/row_demos -d demos/row_demos -t row --stack_snapshot_file logs/finetuned_models/base_stack_finetune_row.pth \
     --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_row.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_row.pth \
     --write_embed --depth_channels_history --cycle_consistency
     
    # stack
    python3 evaluate_demo_correspondence.py -e demos/stack_demos -d demos/stack_demos -t stack --row_snapshot_file logs/finetuned_models/base_row_finetune_stack.pth \
     --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_stack.pth \
     --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_stack.pth --write_embed --depth_channels_history --cycle_consistency

    # unstack
    python3 evaluate_demo_correspondence.py -e demos/unstacking_demos -d demos/unstacking_demos -t unstack --stack_snapshot_file \
     logs/finetuned_models/base_stack_finetune_unstack.pth --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_unstack.pth \
     --row_snapshot_file logs/finetuned_models/base_row_finetune_unstack.pth \
     --write_embed --depth_channels_history --cycle_consistency

    # vertical_square
    python3 evaluate_demo_correspondence.py -e demos/vertical_square_demos -d demos/vertical_square_demos -t vertical_square --row_snapshot_file \
     logs/finetuned_models/base_row_finetune_vertical_square.pth --stack_snapshot_file logs/finetuned_models/base_stack_finetune_vertical_square.pth \
     --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_vertical_square.pth --write_embed --depth_channels_history --cycle_consistency
fi

if [ $ssr -eq 1 ]
then
    # NOTE change ports here AND in commands below if sims need to be run on different ports. 19997-20000 used by default
    $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt &
    $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19998_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt &
    $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt &
    $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_20000_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt &

    # row
    export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19997 --random_seed 1238 \
    --max_test_trials 50 --task_type row --is_testing --use_demo --demo_path demos/row_demos --stack_snapshot_file logs/finetuned_models/base_stack_finetune_row.pth \
    --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_row.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_row.pth \
    --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65 &

    # stack
    export CUDA_VISIBLE_DEVICES="1" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19998 --random_seed 1238 \
    --max_test_trials 50 --task_type stack --is_testing --use_demo --demo_path demos/stack_demos --row_snapshot_file logs/finetuned_models/base_row_finetune_stack.pth \
    --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_stack.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_stack.pth \
    --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65 &

    # unstack
    export CUDA_VISIBLE_DEVICES="2" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19999 --random_seed 1238 \
    --max_test_trials 50 --task_type unstack --is_testing --use_demo --demo_path demos/unstacking_demos --stack_snapshot_file \
    logs/finetuned_models/base_stack_finetune_unstack.pth --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_unstack.pth \
    --row_snapshot_file logs/finetuned_models/base_row_finetune_unstack.pth --grasp_only --depth_channels_history --cycle_consistency \
    --no_common_sense_backprop --future_reward_discount 0.65 &

    # vertical square
    export CUDA_VISIBLE_DEVICES="3" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 20000 --random_seed 1238 \
    --max_test_trials 50 --task_type vertical_square --is_testing --use_demo --demo_path demos/vertical_square_demos --stack_snapshot_file \
    logs/finetuned_models/base_stack_finetune_vertical_square.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_vertical_square.pth \
    --row_snapshot_file logs/finetuned_models/base_row_finetune_vertical_square.pth --grasp_only --depth_channels_history --cycle_consistency \
    --no_common_sense_backprop --future_reward_discount 0.65 &
fi
