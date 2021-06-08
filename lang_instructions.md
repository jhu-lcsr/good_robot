# Language Instructions

1. Download language models here: https://drive.google.com/file/d/14typT6NTc25KjOybmAyS5Pd_0uBVFiew/view?usp=sharing
- Both models pre-trained on state, fine-tuned on all images.
- Unpack models `tar -xzvf models.tar.gz`

2. Move transformer models to desired location and edit config
The config file needs to be edited once you've moved the models. Specifically, there is a line

`checkpoint_dir: /srv/local1/estengel/models/gr_real_data/stacks_ablation/100` 

which needs to be changed to 

`checkpoint_dir: <PATH_TO_MODEL>`

3. Download Q-learning models: 
- Stacking: https://drive.google.com/file/d/1Qga9m4aVxLAGlZ0girFrEG2JrMl-Fy50/view?usp=sharing
- Row-making: https://drive.google.com/file/d/1Nj3kUM4kMe9JbcgPKV-NUY9ZGZblq63s/view?usp=sharing


4. Run with following command: 
- Stacking: 

```export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --is_testing --obj_mesh_dir objects/blocks --num_obj 8 --common_sense  --place  --static_language_mask --language_model_config <PATH_TO_MODEL>/config.yaml --language_model_weights <PATH_TO_MODEL>/best.th --snapshot_file <PATH_TO_RL_MODEL>/2020-06-07-21-42-16_Sim-Stack-SPOT-Trial-Reward-Masked-Training-best-model-good-robot-paper/snapshot.reinforcement_trial_success_rate_best_index.pth --goal_num_obj=4 --end_on_incorrect_order```

- Row-making: 

```export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --is_testing --obj_mesh_dir objects/blocks --num_obj 4   --static_language_mask --language_model_config <PATH_TO_MODEL>/config.yaml --language_model_weights <PATH_TO_MODEL>/best.th --snapshot_file <PATH_TO_RL_MODEL>/2020-06-03-12-05-28_Sim-Rows-Two-Step-Reward-Masked-Training-best-model/snapshot.reinforcement_trial_success_rate_best_index.pth --end_on_incorrect_order --check_row --timeout 100000  --place --separation_threshold 0.08 --distance_threshold 0.04```
