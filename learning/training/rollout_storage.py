import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    @classmethod
    def from_rollouts(cls, rollouts, device=None, intrinsic_reward_only=False):
        # Count how many steps in the rollouts
        num_steps = 0
        for r in rollouts:
            num_steps += len(r)

        # We don't have a hidden state
        rollout_storage = RolloutStorage(num_steps, 1, action_shape=4,
                                         recurrent_hidden_state_size=1)
        print("RolloutStorage: initialize")
        rollout_storage.initialize(rollouts, intrinsic_reward_only=intrinsic_reward_only)
        if device is not None:
            rollout_storage.to(device)

        return rollout_storage

    def __init__(self, num_steps, num_processes, action_shape,
                 recurrent_hidden_state_size):
        #self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.obs = [None]*num_steps
        self.obsb = [None]*num_steps#torch.zeros(num_steps + 1, num_processes, *obsb_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    # Added this function as a bridge between our implementation and Ilya's
    def initialize(self, rollouts, intrinsic_reward_only=False):
        for rollout in rollouts:
            for sample in rollout:
                # Observation at start of timestep t
                # Calling clone here resulted in an un-debuggable hang! Can't figure out why.
                #x = sample["policy_input"].clone()
                self.obs[self.step] = sample["policy_input"]
                #x = sample["policy_input_b"].clone()
                self.obsb[self.step] = sample["policy_input_b"]
                # Action taken at timestep t
                self.actions[self.step].copy_(torch.from_numpy(sample["action"]))
                self.action_log_probs[self.step].copy_(sample["action_logprob"])

                # Predicted value for observation at timestep t
                self.value_preds[self.step].copy_(sample["value_pred"])
                # Reward for executing action a_t at timestep t and ending up at state s_t+1.
                # Really computed based on state s_t+1.
                # TODO: Compute map coverage based on next state too (early-forward stage and keep results)
                if intrinsic_reward_only:
                    self.rewards[self.step] = sample["intrinsic_reward"]
                else:
                    self.rewards[self.step] = sample["full_reward"]

                # Whether timestep t is the last timestep of an episode
                self.masks[self.step] = int(sample["done"])
                self.bad_masks[self.step] = int(sample["expired"])

                self.step += 1

    def to(self, device):
        self.obs = [o.to(device) for o in self.obs if o is not None]
        self.obsb = [o.to(device) for o in self.obsb if o is not None]
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    """
    def insert(self, obs, obsb, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        # TODO: Why is this put in self.step + 1? WTF
        self.obs[self.step + 1].copy_(obs)
        self.obsb[self.step + 1].copy_(obsb)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        # These are put in self.step + 1, because the recurrent policies
        # use these masks to decide when to reinitialize
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps
    """

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        # TODO: Change interpretations of masks as done masks. Thus far they been used as not-done masks. Done for last block
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                #self.value_preds[-1] = next_value
                gae = 0

                # Cumuative returns are the correct empirical returns at the end of every step
                #cumulative_returns = [self.rewards[0]]
                #for step in range(1, self.rewards.size(0)):
                #    cumret = cumulative_returns[step-1] * (1 - self.masks[step-1])
                #    cumulative_returns.append(cumret)

                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * (1 - self.masks[step]) - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * (1 - self.masks[step]) * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                for step in reversed(range(self.rewards.size(0))):
                    # TODO WARNING: Changed here from self.masks[step + 1] to self.masks[step]
                    # TODO: Do the same when using GAE
                    prop = self.returns[step + 1] * gamma * (1 - self.masks[step])
                    ret = prop + self.rewards[step]
                    self.returns[step] = ret

        return self

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False)
        for indices in sampler:
            obs_batch = [self.obs[i] for i in indices]
            obsb_batch = [self.obsb[i] for i in indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            # Remove the redundant dimension - it clashes with things! This should be a 1D vector
            old_action_log_probs_batch = old_action_log_probs_batch[:, 0]

            # TODO: return map coverages / semantic maps or wahtever
            yield obs_batch, obsb_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
