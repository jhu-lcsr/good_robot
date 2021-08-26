#!/bin/sh

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

# regenerate embedding pickles

# row
python3 evaluate_demo_correspondence.py -e demos/row_demos -d demos/row_demos -t row --stack_snapshot_file logs/finetuned_models/base_stack_finetune_row.pth
 --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_row.pth --unstack_snapshot_file base_models/base_unstack_finetune_row.pth --write_embed
 --depth_channels_history --cycle_consistency
 
# stack
python3 evaluate_demo_correspondence.py -e demos/stack_demos -d demos/stack_demos -t stack --row_snapshot_file logs/finetuned_models/base_row_finetune_stack.pth
 --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_stack.pth --unstack.pth_snapshot_file
logs/finetuned_models/base_unstack.pth_finetune_stack.pth --write_embed --depth_channels_history --cycle_consistency

# unstack
python3 evaluate_demo_correspondence.py -e demos/unstack_demos -d demos/unstack_demos -t unstack --stack_snapshot_file logs/finetuned_models/base_stack_finetune_unstack.pth
 --vertical_square_snapshot_file logs/finetuned_models/base_vertical_square_finetune_unstack.pth --row_snapshot_file logs/finetuned_models/base_row_finetune_unstack.pth
 --write_embed --depth_channels_history --cycle_consistency

# vertical_square
python3 evaluate_demo_correspondence.py -e demos/vertical_square_demos -d demos/vertical_square_demos -t vertical_square --row_snapshot_file
logs/finetuned_models/base_row_finetune_vertical_square.pth --stack_snapshot_file logs/finetuned_models/base_stack_finetune_vertical_square.pth
 --unstack_snapshot_file logs/finetuned_models/base_unstack_finetune_vertical_square.pth --write_embed --depth_channels_history --cycle_consistency
