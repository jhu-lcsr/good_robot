# NOTE This is what to modify if your paths are different.
PATH_TO_COPPELIA_SIM="$HOME/src/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/";
PATH_TO_RGR="$HOME/src/real_good_robot";
task="";
coppelia=0;
# TODO implement coppelia flag

# help function
Help()
{
   # Display Help
   echo "Syntax: ./ssr.sh [-t|h]"
   echo "options:"
   echo "-t     Which task? [row|stack|unstack|vertical_square]"
   echo "-c     launch coppeliasim"
   echo "-h     Print help menu."
   echo
   echo "Note that running without any of the options set will result in no action."
}

# get cmd line options
while getopts "t:hc" option; do
    case $option in
        t) # which task?
            task=${OPTARG};;
        h) # help
            Help
            exit;;
        c) # launch coppelia_sim
            coppelia=1;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

if [ "$task" = "row" ]
then
    # NOTE change ports here AND in commands below if sims need to be run on different ports. 19997-20000 used by default
    # $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt

    # row
    export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19997 --random_seed 1238 \
    --max_test_trials 50 --task_type row --is_testing --use_demo --demo_path demos/row_demos --stack_snapshot_file logs/finetuned_models/base_stack_finetune_row.pth \
    --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_row.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_row.pth \
    --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65

elif [ "$task" = "stack" ]
then
    # $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19998_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt &
    # stack
    export CUDA_VISIBLE_DEVICES="1" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19998 --random_seed 1238 \
    --max_test_trials 50 --task_type stack --is_testing --use_demo --demo_path demos/stack_demos --row_snapshot_file logs/finetuned_models/base_row_finetune_stack.pth \
    --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_stack.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_stack.pth \
    --grasp_only --depth_channels_history --cycle_consistency --no_common_sense_backprop --future_reward_discount 0.65


elif [ "$task" = "unstack" ]
then
    # $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt &
    # unstack
    export CUDA_VISIBLE_DEVICES="2" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 19999 --random_seed 1238 \
    --max_test_trials 50 --task_type unstack --is_testing --use_demo --demo_path demos/unstacking_demos --stack_snapshot_file \
    logs/finetuned_models/base_stack_finetune_unstack.pth --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_unstack.pth \
    --row_snapshot_file logs/finetuned_models/base_row_finetune_unstack.pth --grasp_only --depth_channels_history --cycle_consistency \
    --no_common_sense_backprop --future_reward_discount 0.65

elif [ "$task" = "vertical_square" ]
then
    # $PATH_TO_COPPELIA_SIM/coppeliaSim.sh -gREMOTEAPISERVERSERVICE_20000_FALSE_TRUE -s $PATH_TO_RGR/simulation/simulation.ttt &
    # vertical square
    export CUDA_VISIBLE_DEVICES="3" && python3 main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 4 --common_sense --place --tcp_port 20000 --random_seed 1238 \
    --max_test_trials 50 --task_type vertical_square --is_testing --use_demo --demo_path demos/vertical_square_demos --stack_snapshot_file \
    logs/finetuned_models/base_stack_finetune_vertical_square.pth --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_vertical_square.pth \
    --row_snapshot_file logs/finetuned_models/base_row_finetune_vertical_square.pth --grasp_only --depth_channels_history --cycle_consistency \
    --no_common_sense_backprop --future_reward_discount 0.65

else
    echo "Must pass one of [row | stack | unstack | vertical_square] to -t."
fi
