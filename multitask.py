from train import PolicyManagerBase
from utils.visualization_tools import *
import torch
from utils.utils import *
from utils.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baseline multitask learning approach - single policy for all tasks
class Multitask(PolicyManagerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Generate actions using single shared policy (no skill decomposition)
    def get_actions(self, batch, network_bundle, sample_traj, robot_states, old_concatenated_traj):
        b, t, _ = batch['actions'].shape
        if not self.args.image_to_subpolicy:
            raise Exception("should be using image input to subpolicy")
        else:
            subpolicy_input = sample_traj
        # Use zero parameters since there's no skill decomposition
        continuous_param = torch.zeros((b, self.args.traj_length, self.args.z_dimensions)).to(device)
        skill_tensor = torch.zeros((b, self.args.traj_length, 1)).type(torch.int64).to(device)
        task_idx = self.get_one_hot_task_tensor(batch)
        selected_a = network_bundle.primitive_policy_set[0].forward(subpolicy_input, continuous_param, skill_tensor, task_indices = task_idx)

        return {'output_actions': selected_a}

    def calculate_cloning_loss(self, batch, network_bundle, sample_traj, robot_states, old_concatenated_traj):
        act_dict = self.get_actions(batch, network_bundle, sample_traj, robot_states, old_concatenated_traj)
        loss = self.get_prediction_loss(act_dict['output_actions'], batch['actions'], batch['task_length'])
        subpolicy_loss = loss.sum() / sum(batch['task_length'])
        return subpolicy_loss

    def update_policies(self, batch, network_bundle, act_dict):
        # Unpack act_dict
        output_actions = act_dict['output_actions']
        # calculate reconstruction losses
        loss = self.get_prediction_loss(act_dict['output_actions'], batch['actions'], batch['task_length'])

        #variance = self.get_masked_variance(output_actions, batch['task_length'])
        #mse_loss = self.get_masked_mse_loss(output_actions, batch['actions'], batch['task_length'])
        subpolicy_loss = loss.sum() / sum(batch['task_length'])
        #mse_loss = mse_loss.sum() / (self.args.action_dim * sum(batch['task_length']))
        #avg_variance = variance.sum() / ((self.args.action_dim - 1) * sum(batch['task_length']))

        network_bundle.optimizer.zero_grad()
        subpolicy_loss.backward()
        network_bundle.optimizer.step()
        network_bundle.optimizer.zero_grad()

        self.upload_log({"policy loss": torch.mean(subpolicy_loss).cpu().detach().numpy()}, step=self.counter)
                         #"train mse loss": mse_loss}
                         #"gmm avg variance": avg_variance.detach()}

    def visualize_learned_skills(self, network_bundle):
        pass

    def update_policies_downstream(self, network_bundle, task_bundle, batch, act_dict):
        output_actions = act_dict['output_actions']
        #output_actions_valid = act_dict_valid['output_actions']
        # calculate reconstruction losses
        loss = self.get_prediction_loss(output_actions, batch['actions'], batch['task_length'])
        #variance = self.get_masked_variance(output_actions, batch['task_length'])
        #loss_valid = self.get_prediction_loss(output_actions_valid, valid_batch['actions'], valid_batch['task_length'])
        subpolicy_loss = loss.sum() / sum(batch['task_length'])
        #subpolicy_loss_valid = loss_valid.sum() / sum(valid_batch['task_length'])
        #avg_variance = variance.sum() / ((self.args.action_dim - 1) * sum(batch['task_length']))
        total_loss = subpolicy_loss #+ -1*self.args.variance_reg_weight * avg_variance
        network_bundle.optimizer.zero_grad()
        total_loss.backward()
        network_bundle.optimizer.step()
        network_bundle.optimizer.zero_grad()
        Logger.log({f"Task {task_bundle.task_idx} logp loss": subpolicy_loss,
                   #f"Task {task_bundle.task_idx} logp loss validation": subpolicy_loss_valid,
                   #f"Task {task_bundle.task_idx} validation loss ratio": subpolicy_loss_valid / subpolicy_loss,
                   f"Task {task_bundle.task_idx} counter": task_bundle.counter})
                   #f"Task {task_bundle.task_idx} avg variance": avg_variance.detach()})

    # Get action for rollout (execution) using the single multitask policy
    def get_rollout_action(self, network_bundle, sample_traj, robot_state, task_index):
        b, t, _ = sample_traj.shape
        subpolicy_input = sample_traj if self.args.image_to_subpolicy else robot_state
        z = torch.zeros((b, t, self.args.z_dimensions)).to(device)  # No skill parameters
        k = torch.zeros((b, t, 1)).type(torch.int64).to(device)  # No skill selection
        selected_a = network_bundle.primitive_policy_set[0].get_actions(subpolicy_input, z, k, task_indices = task_index)
        return torch.flatten(selected_a[:, -1]).cpu().detach().numpy()



