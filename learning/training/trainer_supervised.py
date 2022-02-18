import sys

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from data_io.instructions import get_all_instructions
from data_io.instructions import get_word_to_token_map
from utils.simple_profiler import SimpleProfiler
from learning.utils import get_n_params, get_n_trainable_params

from parameters.parameter_server import get_current_parameters

PROFILE = False


class Trainer:
    def __init__(
            self,
            model,
            state=None,
            epoch=0,
            name="",
            run_name="",
    ):
        _, _, _, corpus = get_all_instructions()
        self.token2word, self.word2token = get_word_to_token_map(corpus)

        self.params = get_current_parameters()["Training"]
        self.batch_size = self.params['batch_size']
        self.weight_decay = self.params['weight_decay']
        self.optimizer = self.params['optimizer']
        self.num_loaders = self.params['num_loaders']
        self.lr = self.params['lr']

        self.name = name
        self.dataset_names = None

        n_params = get_n_params(model)
        n_params_tr = get_n_trainable_params(model)
        print("Training Model:")
        print("Number of model parameters: " + str(n_params))
        print("Trainable model parameters: " + str(n_params_tr))

        self.model = model
        self.run_name = run_name
        if self.optimizer == "adam":
            self.optim = optim.Adam(self.get_model_parameters(self.model), self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            self.optim = optim.SGD(self.get_model_parameters(self.model), self.lr, weight_decay=self.weight_decay, momentum=0.9)
        self.train_epoch_num = epoch
        self.train_segment = 0
        self.test_epoch_num = epoch
        self.test_segment = 0
        self.set_state(state)
        self.batch_num = 0

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
        state["name"] = self.name
        state["train_epoch_num"] = self.train_epoch_num
        state["train_segment"] = self.train_segment
        state["test_epoch_num"] = self.test_epoch_num
        state["test_segment"] = self.test_segment
        return state

    def set_state(self, state):
        if state is None:
            return
        self.name = state["name"]
        self.train_epoch_num = state["train_epoch_num"]
        self.train_segment = state["train_segment"]
        self.test_epoch_num = state["test_epoch_num"]
        self.test_segment = state["test_segment"]

    def write_grad_summaries(self, writer, named_params, idx):
        for name, parameter in named_params:
            weights = parameter.data.cpu()
            mean_weight = torch.mean(weights)
            weights = weights.numpy()
            writer.add_histogram(self.model.model_name + "_internals" + "/hist_" + name + "_data", weights, idx, bins=100)
            writer.add_scalar(self.model.model_name + "_internals" + "/mean_" + name + "_data", mean_weight, idx)
            if parameter.grad is not None:
                grad = parameter.grad.data.cpu()
                mean_grad = torch.mean(grad)
                grad = grad.numpy()
                writer.add_histogram(self.model.model_name + "_internals" + "/hist_" + name + "_grad", grad, idx, bins=100)
                writer.add_scalar(self.model.model_name + "_internals" + "/mean_" + name + "_grad", mean_grad, idx)

    def write_grouped_loss_summaries(self, writer, losses, idx):
        pass

    def set_dataset_names(self, dataset_names):
        self.dataset_names = dataset_names

    def train_epoch(self, train_data=None, train_envs=None, eval=False):
        if eval:
            self.model.eval()
            inference_type = "eval"
            epoch_num = self.train_epoch_num
            self.test_epoch_num += 1
        else:
            self.model.train()
            inference_type = "train"
            epoch_num = self.train_epoch_num
            self.train_epoch_num += 1

        if self.dataset_names is None:
            dataset_names = get_current_parameters().get("Data").get("dataset_names")
        else:
            dataset_names = self.dataset_names

        print("WARNING: ASSUMING THAT SUPERVISED SINGLE_DOMAIN DATA COMES FROM SIMULATOR!")
        dataset = self.model.get_dataset(data=train_data, envs=train_envs, domain="sim", dataset_names=dataset_names, dataset_prefix="supervised", eval=eval)
        # TODO: Get rid of this:
        if hasattr(dataset, "set_word2token"):
            dataset.set_word2token(self.token2word, self.word2token)

        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_loaders,
            pin_memory=False,
            timeout=0,
            drop_last=False)

        num_samples = len(dataset)
        if num_samples == 0:
            print ("DATASET HAS NO DATA!")
            return -1.0

        num_batches = int((num_samples + self.batch_size - 1) / self.batch_size)

        epoch_loss = 0
        count = 0

        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        prof.tick("out")

        for batch in dataloader:

            if batch is None:
                #print("None batch!")
                continue

            prof.tick("batch_load")
            # Zero gradients before each segment and initialize zero segment loss
            self.optim.zero_grad()

            batch_loss, tensor_store = self.model.sup_loss_on_batch(batch, eval)

            if type(batch_loss) == int:
                print("Ding")

            prof.tick("forward")

            # Backprop and step
            if not eval:
                batch_loss.backward()

                prof.tick("backward")

                # This is SLOW! Don't do it often
                # TODO: Get rid of tensorboard
                #if self.batch_num % 20 == 0 and hasattr(self.model, "writer"):
                #    params = self.model.named_parameters()
                #    self.write_grad_summaries(self.model.writer, params, self.batch_num)

                self.batch_num += 1
                self.optim.step()

                prof.tick("optim")

            # Get losses as floats
            epoch_loss += batch_loss.data.item()
            count += 1

            sys.stdout.write(
                "\r Batch:" + str(count) + " / " + str(num_batches) + " loss: " + str(batch_loss.data.item()))
            sys.stdout.flush()

            self.train_segment += 0 if eval else 1
            self.test_segment += 1 if eval else 0

            prof.tick("rep")

            prof.loop()
            prof.print_stats(10)

        if hasattr(self.model, "write_eoe_summaries"):
            self.model.write_eoe_summaries(inference_type, epoch_num)

        print("")
        epoch_loss /= (count + 1e-15)

        if hasattr(self.model, "writer"):
            self.model.writer.add_scalar(self.name + "/" + inference_type + "_epoch_loss", epoch_loss, epoch_num)

        return epoch_loss
