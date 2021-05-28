import sys
import math
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from learning.modules.key_tensor_store import KeyTensorStore

from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map
from data_io.paths import get_logging_dir
from utils.simple_profiler import SimpleProfiler
from learning.utils import get_n_params, get_n_trainable_params
from learning.dual_dataloader import DualDataloader

from utils.logging_summary_writer import LoggingSummaryWriter
from parameters.parameter_server import get_current_parameters


PROFILE = False


class TrainerBidomainBidata:
    def __init__(
            self,
            model_real,
            model_sim,
            model_critic,
            model_oracle_critic=None,
            state=None,
            epoch=0
    ):
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)

        self.params = get_current_parameters()["Training"]
        self.run_name = get_current_parameters()["Setup"]["run_name"]
        self.batch_size = self.params['batch_size']
        self.iterations_per_epoch = self.params.get("iterations_per_epoch", None)
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer']
        self.critic_loaders = self.params['critic_loaders']
        self.model_common_loaders = self.params['model_common_loaders']
        self.model_sim_loaders = self.params['model_sim_loaders']
        self.lr = self.params['lr']
        self.critic_steps = self.params['critic_steps']
        self.model_steps = self.params['model_steps']
        self.critic_batch_size = self.params["critic_batch_size"]
        self.model_batch_size = self.params["model_batch_size"]
        self.disable_wloss = self.params["disable_wloss"]
        self.sim_steps_per_real_step = self.params.get("sim_steps_per_real_step", 1)

        self.real_dataset_names = self.params.get("real_dataset_names")
        self.sim_dataset_names = self.params.get("sim_dataset_names")

        self.bidata = self.params.get("bidata", False)
        self.sim_steps_per_common_step = self.params.get("sim_steps_per_common_step", 1)

        n_params_real = get_n_params(model_real)
        n_params_real_tr = get_n_trainable_params(model_real)
        n_params_sim = get_n_params(model_sim)
        n_params_sim_tr = get_n_trainable_params(model_sim)
        n_params_c = get_n_params(model_critic)
        n_params_c_tr = get_n_params(model_critic)

        print("Training Model:")
        print("Real # model parameters: " + str(n_params_real))
        print("Real # trainable parameters: " + str(n_params_real_tr))
        print("Sim  # model parameters: " + str(n_params_sim))
        print("Sim  # trainable parameters: " + str(n_params_sim_tr))
        print("Critic  # model parameters: " + str(n_params_c))
        print("Critic  # trainable parameters: " + str(n_params_c_tr))

        # Share those modules that are to be shared between real and sim models
        if not self.params.get("disable_domain_weight_sharing"):
            print("Sharing weights between sim and real modules")
            model_real.steal_cross_domain_modules(model_sim)
        else:
            print("NOT Sharing weights between sim and real modules")

        self.model_real = model_real
        self.model_sim = model_sim
        self.model_critic = model_critic
        self.model_oracle_critic = model_oracle_critic
        if self.model_oracle_critic:
            print("Using oracle critic")

        if self.optimizer == "adam":
            Optim = optim.Adam
        elif self.optimizer == "sgd":
            Optim = optim.SGD
        else:
            raise ValueError(f"Unsuppored optimizer {self.optimizer}")

        self.optim_models = Optim(self.model_real.both_domain_parameters(self.model_sim), self.lr, weight_decay=self.weight_decay)
        self.optim_critic = Optim(self.critic_parameters(), self.lr, weight_decay=self.weight_decay)

        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.set_state(state)

    def set_dataset_names(self, sim_datasets, real_datasets):
        self.sim_dataset_names = sim_datasets
        self.real_dataset_names = real_datasets

    def set_start_epoch(self, epoch):
        self.train_epoch_num = epoch
        self.test_epoch_num = epoch

    def critic_parameters(self):
        for p in self.get_model_parameters(self.model_critic):
            yield p
        if self.model_oracle_critic:
            for p in self.get_model_parameters(self.model_oracle_critic):
                yield p

    def get_model_parameters(self, model):
        params_out = []
        skipped_params = 0
        for param in model.parameters():
            if param.requires_grad:
                params_out.append(param)
            else:
                skipped_params += 1
        print(str(skipped_params) + " parameters frozen")
        return params_out

    def get_state(self):
        state = {}
        state["train_epoch_num"] = self.train_epoch_num
        state["train_segment"] = self.train_segment
        state["test_epoch_num"] = self.test_epoch_num
        state["test_segment"] = self.test_segment
        return state

    def set_state(self, state):
        if state is None:
            return
        self.train_epoch_num = state["train_epoch_num"]
        self.train_segment = state["train_segment"]
        self.test_epoch_num = state["test_epoch_num"]
        self.test_segment = state["test_segment"]

    def train_epoch(self, env_list_common=None, env_list_sim=None, data_list_real=None, data_list_sim=None, eval=False):

        if eval:
            self.model_real.eval()
            self.model_sim.eval()
            self.model_critic.eval()
            inference_type = "eval"
            epoch_num = self.train_epoch_num
            self.test_epoch_num += 1
        else:
            self.model_real.train()
            self.model_sim.train()
            self.model_critic.train()
            inference_type = "train"
            epoch_num = self.train_epoch_num
            self.train_epoch_num += 1

        dataset_real_common = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="real", dataset_names=self.real_dataset_names, dataset_prefix="supervised", eval=eval)
        dataset_sim_common = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="sim", dataset_names=self.sim_dataset_names, dataset_prefix="supervised", eval=eval)
        dataset_real_halfway = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="real", dataset_names=self.real_dataset_names, dataset_prefix="supervised", eval=eval, halfway_only=True)
        dataset_sim_halfway = self.model_real.get_dataset(data=data_list_real, envs=env_list_common, domain="sim", dataset_names=self.sim_dataset_names, dataset_prefix="supervised", eval=eval, halfway_only=True)
        dataset_sim = self.model_sim.get_dataset(data=data_list_sim, envs=env_list_sim, domain="sim", dataset_names=self.sim_dataset_names, dataset_prefix="supervised", eval=eval)

        print("Beginning supervised epoch:")
        print("   Sim dataset names: ", self.sim_dataset_names)
        print("   Dataset sizes: ")
        print("   dataset_real_common  ", len(dataset_real_common))
        print("   dataset_sim_common  ", len(dataset_sim_common))
        print("   dataset_real_halfway  ", len(dataset_real_halfway))
        print("   dataset_sim_halfway  ", len(dataset_sim_halfway))
        print("   dataset_sim  ", len(dataset_sim))
        print("   env_list_sim_len ", len(env_list_sim))
        print("   env_list_common_len ", len(env_list_common))
        if len(dataset_sim) == 0 or len(dataset_sim_common) == 0:
            print("Missing data! Waiting for RL to generate it?")
            return 0

        dual_model_loader = DualDataloader(
            dataset_a=dataset_real_common,
            dataset_b=dataset_sim_common,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.model_common_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="max"
        )

        sim_loader = DataLoader(
            dataset=dataset_sim,
            collate_fn=dataset_sim.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.model_sim_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False
        )
        sim_iterator = iter(sim_loader)

        dual_critic_loader = DualDataloader(
            dataset_a=dataset_real_halfway,
            dataset_b=dataset_sim_halfway,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.critic_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="infinite"
        )
        dual_critic_iterator = iter(dual_critic_loader)

        wloss_before_updates_writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{self.run_name}/discriminator_before_updates")
        wloss_after_updates_writer = LoggingSummaryWriter(log_dir=f"{get_logging_dir()}/runs/{self.run_name}/discriminator_after_updates")

        samples_real = len(dataset_real_common)
        samples_common = len(dataset_sim_common)
        samples_sim = len(dataset_sim)
        if samples_real == 0 or samples_sim == 0 or samples_common == 0:
            print (f"DATASET HAS NO DATA: REAL: {samples_real > 0}, SIM: {samples_sim > 0}, COMMON: {samples_common}")
            return -1.0

        num_batches = len(dual_model_loader)

        epoch_loss = 0
        count = 0
        critic_elapsed_iterations = 0

        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        prof.tick("out")

        # Alternate training critic and model
        for real_batch, sim_batch in dual_model_loader:
            if real_batch is None or sim_batch is None:
                print("none")
                continue

            prof.tick("load_model_data")
            # Train the model for model_steps in a row, then train the critic, and repeat
            critic_batch_num = 0

            if count % self.model_steps == 0 and not eval and not self.disable_wloss:
                #print("\nTraining critic\n")
                # Train the critic for self.critic_steps steps
                for cstep in range(self.critic_steps):
                    # Each batch is actually a single rollout (we batch the rollout data across the sequence)
                    # To collect a batch of rollouts, we need to keep iterating
                    real_store = KeyTensorStore()
                    sim_store = KeyTensorStore()
                    for b in range(self.critic_batch_size):
                        # Get the next non-None batch
                        real_c_batch, sim_c_batch = None, None
                        while real_c_batch is None or sim_c_batch is None:
                            real_c_batch, sim_c_batch = next(dual_critic_iterator)
                        prof.tick("critic_load_data")
                        # When training the critic, we don't backprop into the model, so we don't need gradients here
                        with torch.no_grad():
                            real_loss, real_store_b = self.model_real.sup_loss_on_batch(real_c_batch, eval=eval, halfway=True)
                            sim_loss, sim_store_b = self.model_sim.sup_loss_on_batch(sim_c_batch, eval=eval, halfway=True)
                        prof.tick("critic_features")
                        real_store.append(real_store_b)
                        sim_store.append(sim_store_b)
                        prof.tick("critic_store_append")

                    # Forward the critic
                    # The real_store and sim_store should really be a batch of multiple rollouts
                    wdist_loss_a, critic_store = self.model_critic.calc_domain_loss(real_store, sim_store)

                    prof.tick("critic_domain_loss")

                    # Store the first and last critic loss
                    if cstep == 0:
                        wdist_loss_before_updates = wdist_loss_a.detach().cpu()
                    if cstep == self.critic_steps - 1:
                        wdist_loss_after_updates = wdist_loss_a.detach().cpu()

                    if self.model_oracle_critic:
                        wdist_loss_oracle, oracle_store = self.model_oracle_critic.calc_domain_loss(real_store, sim_store)
                        wdist_loss_a += wdist_loss_oracle

                    # Update the critic
                    critic_batch_num += 1
                    self.optim_critic.zero_grad()
                    # Wasserstein distance is maximum distance transport cost under Lipschitz constraint, so we maximize it
                    (-wdist_loss_a).backward()
                    self.optim_critic.step()
                    #sys.stdout.write(f"\r    Critic batch: {critic_batch_num}/{critic_steps} d_loss: {wdist_loss_a.data.item()}")
                    #sys.stdout.flush()
                    prof.tick("critic_backward")

                # Write wasserstein loss before and after wasertein loss updates
                prefix = "pvn_critic" + ("/eval" if eval else "/train")
                wloss_before_updates_writer.add_scalar(f"{prefix}/w_score_before_updates", wdist_loss_before_updates.item(), self.model_critic.get_iter())
                wloss_after_updates_writer.add_scalar(f"{prefix}/w_score_before_updates", wdist_loss_after_updates.item(), self.model_critic.get_iter())

                critic_elapsed_iterations += 1

                # Clean up GPU memory
                del wdist_loss_a
                del critic_store
                del real_store
                del sim_store
                del real_store_b
                del sim_store_b
                prof.tick("del")


            # Forward the model on the bi-domain data
            disable_losses = ["visitation_dist"] if self.params.get("disable_real_loss") else []
            real_loss, real_store = self.model_real.sup_loss_on_batch(real_batch, eval, halfway=False, grad_noise=False, disable_losses=disable_losses)
            sim_loss, sim_store = self.model_sim.sup_loss_on_batch(sim_batch, eval, halfway=False)
            prof.tick("model_forward")

            # Forward the model K times on simulation only data
            for b in range(self.sim_steps_per_common_step):
                # Get the next non-None batch
                sim_batch = None
                while sim_batch is None:
                    try:
                        sim_batch = next(sim_iterator)
                    except StopIteration as e:
                        sim_iterator = iter(sim_loader)
                        print("retry")
                        continue
                prof.tick("load_model_data")
                sim_loss_b, _ = self.model_sim.sup_loss_on_batch(sim_batch, eval, halfway=False)
                sim_loss = (sim_loss + sim_loss_b) if sim_loss else sim_loss_b
                #print(f"  Model forward common sim, loss: {sim_loss_b.detach().cpu().item()}")
                prof.tick("model_forward")

            prof.tick("model_forward")

            # TODO: Reconsider this
            sim_loss = sim_loss / max(self.model_batch_size, self.sim_steps_per_common_step)
            real_loss = real_loss / max(self.model_batch_size, self.sim_steps_per_common_step)
            total_loss = real_loss + sim_loss

            if not self.disable_wloss:
                # Forward the critic
                wdist_loss_b, critic_store = self.model_critic.calc_domain_loss(real_store, sim_store)
                # Increase the iteration on the oracle without running it so that the Tensorboard plots align
                if self.model_oracle_critic:
                    self.model_oracle_critic.iter += 1

                prof.tick("model_domain_loss")
                # Minimize average real/sim losses, maximize domain loss
                total_loss = total_loss + wdist_loss_b

            # Grad step
            if not eval and total_loss.requires_grad:
                self.optim_models.zero_grad()
                total_loss.backward()
                self.optim_models.step()
                prof.tick("model_backward")

            print(f"Batch: {count}/{num_batches} r_loss: {real_loss.data.item() if real_loss else None} s_loss: {sim_loss.data.item()}")

            # Get losses as floats
            epoch_loss += total_loss.data.item()
            count += 1

            self.train_segment += 0 if eval else 1
            self.test_segment += 1 if eval else 0

            prof.loop()
            prof.print_stats(self.model_steps)

            if self.iterations_per_epoch and count > self.iterations_per_epoch:
                break

        print("")
        epoch_loss /= (count + 1e-15)

        return epoch_loss
