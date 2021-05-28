from utils.dict_tools import dict_cross_map
import torch.nn as nn

class AuxiliaryLosses(nn.Module):
    def __init__(self):
        super(AuxiliaryLosses, self).__init__()
        self.aux_keys = []
        self.auxiliaries = {}

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        for key, aux in self.auxiliaries.items():
            aux.cuda(device)

    def to(self, device):
        nn.Module.to(self, device)
        for key, aux in self.auxiliaries.items():
            aux.to(device)

    def input_required(self, input_name):
        for aux_key, aux in self.auxiliaries.items():
            if input_name in aux.get_required_inputs():
                return True
        return False

    def add_auxiliary(self, auxiliary_objective, key=None):
        """
        Adds an auxiliary objective, which is a subclass of auxiliary_objective_base
        :param auxiliary_objective:
        :param key:
        :return:
        """
        if key is None:
            key = auxiliary_objective.get_name()
        self.auxiliaries[key] = auxiliary_objective
        self.add_module(key, auxiliary_objective)
        self.aux_keys.append(key)

    def print_auxiliary_info(self):
        print("Using auxiliary objectives:")
        for key in self.auxiliaries:
            print("       - " + key)

    def calculate_aux_loss(self, tensor_store, reduce_average=False, disable_losses=[]):
        """
        Evaluates all auxiliary objectives, taking their inputs from the kept inputs (from keep_input calls)
        Returns their losses in a dictionary
        :param targets: Dict, where keys are auxiliary names and values are lists of labels.
            For each auxiliary, the number of labels provided must match the total number of inputs previously stored
            If a given auxiliary doesn't require a target value, then it's key can be omitted
        :return: Dict, where keys are auxiliary names and values are Variables with the total loss value
        """
        loss_dict = {}
        count_dict = {}
        metric_dict = {}
        metric_count_dict = {}

        disable_losses = set(disable_losses)
        for module_name in self.aux_keys:
            # Skip those losses that have been requested to be disabled
            if module_name in disable_losses:
                #print("Skipping loss: ", module_name)
                continue
            input_names = self.auxiliaries[module_name].get_required_inputs()
            input_list = []
            for input_name in input_names:
                if input_name == "tensor_store":
                    input_list.append(tensor_store)
                else:
                    input_list.append(tensor_store.get(input_name))
            #input_list = list(zip(*input_list))
            # Input list is a list of lists, where outer list is over timesteps and inner list is over inputs to the auxiliary

            if None in input_list:
                print(f"Skipping aux objective: {module_name} due to missing inputs")
                for j,inp in enumerate(input_list):
                    if inp is None:
                        print(f"   {input_names[j]}")
                continue

            try:
                ret_vals = self.auxiliaries[module_name](*input_list)
            except Exception as e:
                print(f"Exception encountered when calling auxiliary objective {module_name}")
                raise e
            if len(ret_vals) == 2:
                loss, count = ret_vals
                metrics = {}
            elif len(ret_vals) == 3:
                loss, metrics, count = ret_vals
            else:
                raise ValueError(f"Auxiliary objective returned {len(ret_vals)} arguments. Expected 2 or 3.")
            if loss is None:
                continue

            if module_name in loss_dict:
                loss_dict[module_name] += loss
                count_dict[module_name] += count
                for k,v in metrics.items():
                    metric_dict[f"{module_name}/{k}"] += v
                    metric_count_dict[f"{module_name}/{k}"] += count
            else:
                loss_dict[module_name] = loss
                count_dict[module_name] = count
                for k,v in metrics.items():
                    metric_dict[f"{module_name}/{k}"] = v
                    metric_count_dict[f"{module_name}/{k}"] = count

        if reduce_average:
            avg_loss_dict = dict_cross_map(loss_dict, count_dict, lambda a, b: a / (b + 1e-9))
            avg_metric_dict = dict_cross_map(metric_dict, metric_count_dict, lambda a, b: a / (b + 1e-9))
            return avg_loss_dict, avg_metric_dict
        else:
            return loss_dict, metric_dict, count_dict

    def combine_losses(self, aux_losses, loss_weights):
        """
        Takes a dictionary of auxiliary losses and a dictionary of associated weights, where weights and losses
        are identified by the keys of the auxiliary objectives from which they came from.
        Outputs a single loss value, which is a convex combination of auxiliary losses with the given weights
        :param aux_losses:
        :param loss_weights:
        :return:
        """
        total_loss = None
        for key in aux_losses:
            weight = 1
            if key in loss_weights:
                weight = loss_weights[key]
            else:
                raise Exception("Auxiliary weight not defined for " + str(key))
            this_loss = aux_losses[key] * weight
            if total_loss is None:
                total_loss = this_loss
            else:
                total_loss += this_loss
        if total_loss is None:
            return 0
        return total_loss
