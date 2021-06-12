import sys
import math

import torch
import torch.optim as optim

from learning.modules.key_tensor_store import KeyTensorStore

from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map
from utils.simple_profiler import SimpleProfiler
from learning.utils import get_n_params, get_n_trainable_params
from learning.dual_dataloader import DualDataloader

from utils.logging_summary_writer import LoggingSummaryWriter
from parameters.parameter_server import get_current_parameters


PROFILE = False


class TrainerBidomain:
    def __init__(
            self,
            model_real,
            model_sim,
            model_critic,
            state=None,
            epoch=0
    ):
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)

        self.params = get_current_parameters()["Training"]
        self.run_name = get_current_parameters()["Setup"]["run_name"]
        self.batch_size = self.params['batch_size']
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer']
        self.num_loaders = self.params['num_loaders']
        self.lr = self.params['lr']
        self.critic_steps = self.params['critic_steps']
        self.critic_warmup_steps = self.params['critic_warmup_steps']
        self.critic_warmup_iterations = self.params['critic_warmup_iterations']
        self.model_steps = self.params['model_steps']
        self.critic_batch_size = self.params["critic_batch_size"]
        self.model_batch_size = self.params["model_batch_size"]
        self.disable_wloss = self.params["disable_wloss"]
        self.sim_steps_per_real_step = self.params.get("sim_steps_per_real_step", 1)
        self.real_grad_noise = self.params.get("real_grad_noise", False)

        self.critic_steps_cycle = self.params.get("critic_steps_cycle", False)
        self.critic_steps_amplitude = self.params.get("critic_steps_amplitude", 0)
        self.critic_steps_period = self.params.get("critic_steps_period", 1)

        self.sim_datasets = get_current_parameters()["Data"]["sim_datasets"]
        self.real_datasets = get_current_parameters()["Data"]["real_datasets"]

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

        if self.optimizer == "adam":
            Optim = optim.Adam
        elif self.optimizer == "sgd":
            Optim = optim.SGD
        else:
            raise ValueError(f"Unsuppored optimizer {self.optimizer}")

        self.optim_models = Optim(self.model_real.both_domain_parameters(self.model_sim), self.lr, weight_decay=self.weight_decay)
        self.optim_critic = Optim(self.get_model_parameters(self.model_critic), self.lr, weight_decay=self.weight_decay)

        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.set_state(state)

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

    def train_epoch(self, env_list=None, data_list_real=None, data_list_sim=None, eval=False, restricted_domain=False):

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

        # Allow testing with both domains being simulation domain
        if self.params["sim_domain_only"]:
            dataset_real = self.model_sim.get_dataset(data=data_list_sim, envs=env_list, dataset_names=self.sim_datasets, dataset_prefix="supervised", eval=eval)
            self.model_real = self.model_sim
        else:
            dataset_real = self.model_real.get_dataset(data=data_list_real, envs=env_list, dataset_names=self.real_datasets, dataset_prefix="supervised", eval=eval)

        dataset_sim = self.model_sim.get_dataset(data=data_list_sim, envs=env_list, dataset_names=self.sim_datasets, dataset_prefix="supervised", eval=eval)

        dual_model_loader = DualDataloader(
            dataset_a=dataset_real,
            dataset_b=dataset_sim,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="max"
        )

        dual_critic_loader = DualDataloader(
            dataset_a=dataset_real,
            dataset_b=dataset_sim,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False,
            joint_length="infinite"
        )
        dual_critic_iterator = iter(dual_critic_loader)

        #wloss_before_updates_writer = LoggingSummaryWriter(log_dir=f"runs/{self.run_name}/discriminator_before_updates")
        #wloss_after_updates_writer = LoggingSummaryWriter(log_dir=f"runs/{self.run_name}/discriminator_after_updates")

        samples_real = len(dataset_real)
        samples_sim = len(dataset_sim)
        if samples_real == 0 or samples_sim == 0:
            print (f"DATASET HAS NO DATA: REAL: {samples_real > 0}, SIM: {samples_sim > 0}")
            return -1.0

        num_batches = len(dual_model_loader)

        epoch_loss = 0
        count = 0
        critic_elapsed_iterations = 0

        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        prof.tick("out")

        # Alternate training critic and model
        for real_batch, sim_batch in dual_model_loader:
            if restricted_domain == "real":
                sim_batch = real_batch
            elif restricted_domain == "simulator":
                real_batch = sim_batch
            if real_batch is None or sim_batch is None:
                continue

            # We run more updates on the sim data than on the real data to speed up training and
            # avoid overfitting on the scarce real data
            if self.sim_steps_per_real_step == 1 or self.sim_steps_per_real_step == 0 or count % self.sim_steps_per_real_step == 0:
                train_sim_only = False
            else:
                train_sim_only = True

            if sim_batch is None or (not train_sim_only and real_batch is None):
                continue

            prof.tick("load_model_data")
            # Train the model for model_steps in a row, then train the critic, and repeat
            critic_batch_num = 0

            if count % self.model_steps == 0 and not eval and not self.disable_wloss:
                #print("\nTraining critic\n")
                # Train the critic for self.critic_steps steps
                if critic_elapsed_iterations > self.critic_warmup_iterations:
                    critic_steps = self.critic_steps
                    if self.critic_steps_cycle:
                        critic_steps_delta = int(self.critic_steps_amplitude * math.sin(count * 3.14159 / self.critic_steps_period) + 0.5)
                        critic_steps += critic_steps_delta
                else:
                    critic_steps = self.critic_warmup_steps

                assert (critic_steps > 0), "Need more than one iteration for critic!"
                for cstep in range(critic_steps):

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
                    #if cstep == 0:
                    #    wdist_loss_before_updates = wdist_loss_a.detach().cpu()
                    #if cstep == critic_steps - 1:
                    #    wdist_loss_after_updates = wdist_loss_a.detach().cpu()

                    # Update the critic
                    critic_batch_num += 1
                    self.optim_critic.zero_grad()
                    # Wasserstein distance is maximum distance transport cost under Lipschitz constraint, so we maximize it
                    (-wdist_loss_a).backward()
                    self.optim_critic.step()
                    sys.stdout.write(f"\r    Critic batch: {critic_batch_num}/{critic_steps} d_loss: {wdist_loss_a.data.item()}")
                    sys.stdout.flush()
                    prof.tick("critic_backward")

                # Write wasserstein loss before and after wasertein loss updates
                #prefix = "pvn_critic" + ("/eval" if eval else "/train")
                #wloss_before_updates_writer.add_scalar(f"{prefix}/w_score_before_updates", wdist_loss_before_updates.item(), self.model_critic.get_iter())
                #wloss_after_updates_writer.add_scalar(f"{prefix}/w_score_before_updates", wdist_loss_after_updates.item(), self.model_critic.get_iter())

                critic_elapsed_iterations += 1
                print("Continuing model training\n")

                # Clean up GPU memory
                del wdist_loss_a
                del critic_store
                del real_store
                del sim_store
                del real_store_b
                del sim_store_b
                prof.tick("del")

            # Forward the model
            real_store = KeyTensorStore()
            sim_store = KeyTensorStore()
            real_loss = None
            sim_loss = None
            # TODO: Get rid of this loop!. It doesn't even loop over and sample new batches
            for b in range(self.model_batch_size):
                real_loss_b, real_store_b = self.model_real.sup_loss_on_batch(real_batch, eval, halfway=train_sim_only, grad_noise=self.real_grad_noise)
                real_loss = (real_loss + real_loss_b) if real_loss else real_loss_b
                real_store.append(real_store_b)

                sim_loss_b, sim_store_b = self.model_sim.sup_loss_on_batch(sim_batch, eval, halfway=False)
                sim_loss = (sim_loss + sim_loss_b) if sim_loss else sim_loss_b
                sim_store.append(sim_store_b)
                prof.tick("model_forward")

            sim_loss = sim_loss / self.model_batch_size
            if train_sim_only:
                total_loss = sim_loss
            else:
                real_loss = real_loss / self.model_batch_size
                total_loss = real_loss + sim_loss

            if not self.disable_wloss:
                # Forward the critic
                wdist_loss_b, critic_store = self.model_critic.calc_domain_loss(real_store, sim_store)

                prof.tick("model_domain_loss")
                # Minimize average real/sim losses, maximize domain loss
                total_loss = total_loss + wdist_loss_b

            # Grad step
            if not eval and total_loss.requires_grad:
                self.optim_models.zero_grad()
                try:
                    total_loss.backward()
                except Exception as e:
                    print("Error backpropping: ")
                    print(e)
                self.optim_models.step()
                prof.tick("model_backward")

            sys.stdout.write(f"\r Batch: {count}/{num_batches} r_loss: {real_loss.data.item() if real_loss else None} s_loss: {sim_loss.data.item()}")
            sys.stdout.flush()

            # Get losses as floats
            epoch_loss += total_loss.data.item()
            count += 1

            self.train_segment += 0 if eval else 1
            self.test_segment += 1 if eval else 0

            prof.loop()
            prof.print_stats(self.model_steps)

        print("")
        epoch_loss /= (count + 1e-15)

        return epoch_loss
