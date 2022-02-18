import numpy as np
import torch
from pprint import pprint

from data_io.paths import get_logging_dir
from data_io.models import load_model
from data_io.train_data import save_dataset

from rollout.simple_parallel_rollout import SimpleParallelPolicyRoller
from rollout.rollout_sampler import RolloutSampler

from learning.training.rollout_storage import RolloutStorage
from learning.training.ppo import PPO
from learning.models.model_pvn_stage1_bidomain import PVN_Stage1_Bidomain
from evaluation.evaluate_nl import DataEvalNL

from itertools import chain

from utils.logging_summary_writer import LoggingSummaryWriter
from utils.dict_tools import dictlist_append, dict_map
from learning.utils import get_n_params, get_n_trainable_params

from utils.simple_profiler import SimpleProfiler
from utils.dict_tools import dict_merge
from transformations import pos_m_to_px

PROFILE = False

# TODO: Move this rollout statistics code to it's permanent home
# ---------------------------------------------------------------------------


def stop_success(rollout):
    last_sample = rollout[-1]
    state = last_sample["state"]
    stop_pos = state.get_pos_2d()
    # TODO: Grab these from parameter serve
    stop_pos_map = pos_m_to_px(stop_pos[np.newaxis, :], img_size_px=32, world_size_px=32, world_size_m=4.7)[0]
    goal_distribution = last_sample["v_dist_w"][1,:,:]

    _, argmax_best_goal = goal_distribution.view(-1).max(0)
    best_stop_pos_x = int(argmax_best_goal / goal_distribution.shape[0])
    best_stop_pos_y = int(argmax_best_goal % goal_distribution.shape[0])

    best_stop_pos = torch.Tensor([best_stop_pos_x, best_stop_pos_y])
    pos = torch.from_numpy(stop_pos_map).float()
    dst_to_best_stop = torch.norm(pos - best_stop_pos)

    return dst_to_best_stop.detach().item() < 3.2


def calc_rollout_metrics(rollouts):
    ev = DataEvalNL("", save_images=False, entire_trajectory=False)
    metrics = {}
    total_task_success = 0
    total_stop_success = 0
    coverages = []
    reward_keys = [k for k in rollouts[0][0].keys() if k.endswith("_reward")]
    rewards = {k:[] for k in reward_keys}

    for rollout in rollouts:
        success = ev.rollout_success(rollout[0]["env_id"], rollout[0]["set_idx"], rollout[0]["seg_idx"], rollout)

        # Take sum of rewards for each type of reward
        for k in reward_keys:
            v = sum([s[k] for s in rollout])
            rewards[k].append(v)

        visit_success = stop_success(rollout)
        total_stop_success += 1 if visit_success else 0
        total_task_success += 1 if success else 0

    task_success_rate = total_task_success / len(rollouts)
    visit_success_rate = total_stop_success / len(rollouts)

    metrics["task_success_rate"] = task_success_rate
    metrics["visit_success_rate"] = visit_success_rate

    rollout_lens = [len(rollout) for rollout in rollouts]
    metrics["mean_rollout_len"] = np.asarray(rollout_lens).mean()

    # Average each reward across rollouts
    rewards = {k:np.mean(np.asarray(l)) for k,l in rewards.items()}
    metrics = dict_merge(metrics, rewards)
    return metrics

# ---------------------------------------------------------------------------


class TrainerRL:
    def __init__(self, params, save_rollouts_to_dataset="", device=None):
        self.iterations_per_epoch = params.get("iterations_per_epoch", 1)
        self.test_iterations_per_epoch = params.get("test_iterations_per_epoch", 1)
        self.num_workers = params.get("num_workers")
        self.num_rollouts_per_iter = params.get("num_rollouts_per_iter")
        self.model_name = params.get("model") or params.get("rl_model")
        self.init_model_file = params.get("model_file")
        self.num_steps = params.get("trajectory_len")
        self.device = device

        self.summary_every_n = params.get("plot_every_n")

        self.roller = SimpleParallelPolicyRoller(
            num_workers=self.num_workers,
            device=self.device,
            policy_name=self.model_name,
            policy_file=self.init_model_file,
            dataset_save_name=save_rollouts_to_dataset)

        self.rollout_sampler = RolloutSampler(self.roller)

        # This should load it's own weights from file based on
        self.full_model, _ = load_model(self.model_name)
        self.full_model = self.full_model.to(self.device)
        self.actor_critic = self.full_model.stage2_action_generation
        # Train in eval mode to disable dropout
        #self.actor_critic.eval()
        self.full_model.stage1_visitation_prediction.eval()
        self.writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{params['run_name']}/ppo")

        self.global_step = 0
        self.stage1_updates = 0

        clip_param = params.get("clip")
        num_mini_batch = params.get("num_mini_batch")
        value_loss_coef = params.get("value_loss_coef")
        lr = params.get("lr")
        eps = params.get("eps")
        max_grad_norm = params.get("max_grad_norm")
        use_clipped_value_loss = params.get("use_clipped_value_loss")

        self.entropy_coef = params.get("entropy_coef")
        self.entropy_schedule_epochs = params.get("entropy_schedule_epochs", [])
        self.entropy_schedule_multipliers = params.get("entropy_schedule_multipliers", [])

        self.minibatch_size = params.get("minibatch_size")

        self.use_gae = params.get("use_gae")
        self.gamma = params.get("gamma")
        self.gae_lambda = params.get("gae_lambda")
        self.intrinsic_reward_only = params.get("intrinsic_reward_only")

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        print(f"PPO trainable parameters: {get_n_trainable_params(self.actor_critic)}")
        print(f"PPO actor-critic all parameters: {get_n_params(self.actor_critic)}")

        self.ppo = PPO(self.actor_critic,
                       clip_param=clip_param,
                       ppo_epoch=1,
                       num_mini_batch=num_mini_batch,
                       value_loss_coef=value_loss_coef,
                       entropy_coef=self.entropy_coef,
                       lr=lr,
                       eps=eps,
                       max_grad_norm=max_grad_norm,
                       use_clipped_value_loss=use_clipped_value_loss)

    def set_start_epoch(self, epoch):
        prints_per_epoch = int(self.iterations_per_epoch / self.summary_every_n)
        self.global_step = epoch * prints_per_epoch

    def save_rollouts(self, rollouts, dataset_name):
        for rollout in rollouts:
            # This saves just a single segment per environment, as opposed to all segments that the oracle saves. Problem?
            if len(rollout) > 0:
                env_id = rollout[0]["env_id"]
                save_dataset(dataset_name, rollout, env_id=env_id, lock=True)

    def reload_stage1(self, module_state_dict):
        print("Reloading stage 1 model in RL trainer")
        self.full_model.stage1_visitation_prediction.load_state_dict(module_state_dict)
        print("Reloading stage 1 model in rollout sampler")
        self.rollout_sampler.update_stage1_on_workers(self.full_model.stage1_visitation_prediction)
        print("Done reloading stage1")
        self.stage1_updates += 1

    def train_epoch(self, epoch_num, eval=False, envs="train"):

        rewards = []
        returns = []
        value_losses = []
        action_losses = []
        dist_entropies = []
        value_preds = []
        vels = []
        stopprobs = []

        step_rollout_metrics = {}

        # Update entropy coefficient by applying scaling
        if len(self.entropy_schedule_epochs ) > 0:
            scaled_entropy_coeff = self.entropy_coef
            for e_multiplier, e_epoch in zip(self.entropy_schedule_multipliers, self.entropy_schedule_epochs):
                if epoch_num > e_epoch:
                    scaled_entropy_coeff = e_multiplier * self.entropy_coef
                else:
                    break
            self.ppo.set_entropy_coef(scaled_entropy_coeff)
        else:
            scaled_entropy_coeff = self.entropy_coef

        self.prof.tick("out")

        # TODO: Make the 100 a parameter
        iterations = self.test_iterations_per_epoch if eval else self.iterations_per_epoch

        for i in range(iterations):
            policy_state = self.full_model.get_policy_state()
            device = policy_state[next((iter(policy_state)))].device
            print("TrainerRL: Sampling N Rollouts")
            rollouts = self.rollout_sampler.sample_n_rollouts(self.num_rollouts_per_iter, policy_state, sample=not eval, envs=envs)
            #if save_rollouts_to_dataset is not None:
            #    self.save_rollouts(rollouts, save_rollouts_to_dataset)

            self.prof.tick("sample_rollouts")
            print("TrainerRL: Calculating Rollout Metrics")
            i_rollout_metrics = calc_rollout_metrics(rollouts)
            step_rollout_metrics = dictlist_append(step_rollout_metrics, i_rollout_metrics)

            assert len(rollouts) > 0

            # Convert our rollouts to the format used by Ilya Kostrikov
            device = next(self.full_model.parameters()).device
            rollout_storage = RolloutStorage.from_rollouts(rollouts, device=device, intrinsic_reward_only=self.intrinsic_reward_only)
            next_value = None

            rollout_storage.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda, False)

            self.prof.tick("compute_storage")

            reward = rollout_storage.rewards.mean().detach().cpu().item()
            avg_return = (((rollout_storage.returns[1:] * rollout_storage.masks[:-1]).sum() + rollout_storage.returns[0]) / (rollout_storage.masks[:-1].sum() + 1)).cpu().item()
            avg_value = rollout_storage.value_preds.mean().detach().cpu().item()
            avg_vel = rollout_storage.actions[:, 0, 0:3].detach().cpu().numpy().mean(axis=0, keepdims=False)
            avg_stopprob = rollout_storage.actions[:, 0, 3].mean().detach().cpu().item()

            print("TrainerRL: PPO Update")
            if not eval:
                value_loss, action_loss, dist_entropy, avg_ratio = self.ppo.update(rollout_storage, self.global_step, self.minibatch_size)
                print(f"Iter: {i}/{iterations}, Value loss: {value_loss}, Action loss: {action_loss}, Entropy: {dist_entropy}, Reward: {reward}")
            else:
                value_loss = 0
                action_loss = 0
                dist_entropy = 0
                avg_ratio = 0

            self.prof.tick("ppo_update")
            print("TrainerRL: PPO Update Done")

            returns.append(avg_return)
            rewards.append(reward)
            value_losses.append(value_loss)
            action_losses.append(action_loss)
            dist_entropies.append(dist_entropy)
            value_preds.append(avg_value)
            vels.append(avg_vel)
            stopprobs.append(avg_stopprob)

            if i % self.summary_every_n == self.summary_every_n - 1:
                avg_reward = np.mean(np.asarray(rewards[-self.summary_every_n:]))
                avg_return = np.mean(np.asarray(returns[-self.summary_every_n:]))
                avg_vel = np.mean(np.asarray(vels[-self.summary_every_n:]), axis=0, keepdims=False)

                metrics = {
                    "value_loss": np.mean(np.asarray(value_losses[-self.summary_every_n:])),
                    "action_loss": np.mean(np.asarray(action_losses[-self.summary_every_n:])),
                    "dist_entropy": np.mean(np.asarray(dist_entropies[-self.summary_every_n:])),
                    "avg_value": np.mean(np.asarray(value_preds[-self.summary_every_n:])),
                    "avg_vel_x": avg_vel[0],
                    "avg_yaw_rate": avg_vel[2],
                    "avg_stopprob": np.mean(np.asarray(stopprobs[-self.summary_every_n:])),
                    "ratio": avg_ratio
                }

                # Reduce average
                step_rollout_metrics = dict_map(step_rollout_metrics, lambda m: np.asarray(m).mean())

                mode = "eval" if eval else "train"

                self.writer.add_scalar(f"ppo_{mode}/reward", avg_reward, self.global_step)
                self.writer.add_scalar(f"ppo_{mode}/return", avg_return, self.global_step)
                self.writer.add_scalar(f"ppo_{mode}/stage1_updates", self.stage1_updates, self.global_step)
                self.writer.add_dict(f"ppo_{mode}/", metrics, self.global_step)
                self.writer.add_dict(f"ppo_{mode}/", step_rollout_metrics, self.global_step)
                self.writer.add_scalar(f"ppo_{mode}/scaled_entropy_coeff", scaled_entropy_coeff, self.global_step)
                step_rollout_metrics = {}

                self.global_step += 1

            self.prof.tick("logging")
            print("TrainerRL: Finished Step")

        # TODO: Remove code duplication (this was easier for now)
        avg_reward = np.mean(np.asarray(rewards))
        avg_vel = np.mean(np.asarray(vels), axis=0, keepdims=False)
        metrics = {
            "value_loss": np.mean(np.asarray(value_losses)),
            "action_loss": np.mean(np.asarray(action_losses)),
            "dist_entropy": np.mean(np.asarray(dist_entropies)),
            "avg_value": np.mean(np.asarray(value_preds)),
            "avg_vel_x": avg_vel[0],
            "avg_yaw_rate": avg_vel[2],
            "avg_stopprob": np.mean(np.asarray(stopprobs))
        }
        #pprint(metrics)

        self.prof.tick("logging")
        self.prof.loop()
        self.prof.print_stats(1)

        return avg_reward, metrics


