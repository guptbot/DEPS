from train import PolicyManagerBase
from utils.visualization_tools import *
import torch
from utils.utils import *
from utils.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DEPS with compression: learns compressed robot state representation
class Deps(PolicyManagerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.number_policies == 1:
            raise Exception("Need to train more than one policy")

    # Generate actions using both variational and latent (online) policies
    def get_actions(self, batch, network_bundle, sample_traj, robot_states, concatenated_traj):
        # Variational policy: infers skills from full trajectory with actions
        # No gradient from the variatinal network to the image encoder, since the variational network is not used at inference time. 
        variational_k, variational_z = network_bundle.variational_policy.forward(
            input=concatenated_traj.detach(), language_input=batch["task_emb"], task_lengths=batch["task_length"])
        
        # Potentially add some noise to the continuous parameter to make the parameter space "smooth"
        if not self.args.add_noise_to_z:
            variational_z_noisy = variational_z
        else:
            variational_z_noisy = variational_z + torch.randn_like(variational_z) / self.args.variance_factor
        selected_a = []
        selected_a_uncorrected = []
        b, t, _ = batch['actions'].shape
        for skill_idx in range(self.args.number_policies):
            skill_tensor = torch.tensor(skill_idx).to(device).reshape(1, 1).repeat(b, t)
            continuous_param = variational_z_noisy[:, skill_idx].reshape(b, 1, self.args.z_dimensions).repeat(1, t, 1)

            # Corrected actions use a residual policy that has access to the full input image. This is not used right now. 
            actions, uncorrected_actions = network_bundle.primitive_policy_set[0].forward(sample_traj, robot_states, continuous_param, skill_tensor)
            selected_a.append(actions)
            selected_a_uncorrected.append(uncorrected_actions)

        # Latent policies: predict skills autoregressively from past observations only
        latent_k = network_bundle.discrete_policy.forward(input=sample_traj.detach(), language=batch['task_emb'])
        latent_z = network_bundle.continuous_policy.forward(input=sample_traj.detach(), language=batch['task_emb'])

        # Get the action predictions from the latent networks. This is used for logging purposes
        selected_a_latent = []
        selected_a_latent_uncorrected = []
        for skill_idx in range(self.args.number_policies):
            skill_tensor = torch.tensor(skill_idx).to(device).reshape(1, 1).repeat(b, t)
            continuous_param = latent_z[:, :, skill_idx]
            actions, uncorrected_actions = network_bundle.primitive_policy_set[0].forward(sample_traj, robot_states, continuous_param, skill_tensor)
            selected_a_latent.append(actions)
            selected_a_latent_uncorrected.append(uncorrected_actions)
        return {'variational_k': variational_k, 'variational_z': variational_z, 'latent_k': latent_k, 'latent_z': latent_z,
                'output_actions': selected_a, 'output_actions_uncorrected': selected_a_uncorrected,
                'output_actions_latent': selected_a_latent, 'output_actions_latent_uncorrected': selected_a_latent_uncorrected}

    def calculate_cloning_loss(self, batch, network_bundle, sample_traj, robot_states, old_concatenated_traj):
        act_dict = self.get_actions(batch, network_bundle, sample_traj, robot_states, old_concatenated_traj)
        output_actions = act_dict['output_actions']

        # Weight the loss for each discrete skill by the probabaility of sampling that discrete skill from the variational network
        weighted_loss = self.calculate_weighted_loss(act_dict['variational_k'],
                                                     lambda i: self.get_prediction_loss(output_actions[i], batch['actions'], batch['task_length']))
        subpolicy_loss = weighted_loss.sum() / sum(batch['task_length'])
        return subpolicy_loss

    # Compute weighted loss across skills using skill probabilities
    def calculate_weighted_loss(self, variational_k, loss_func):
        losses = []
        for skill_idx in range(self.args.number_policies):
            loss = loss_func(skill_idx)
            losses.append(loss)
        losses = torch.stack(losses, dim = -1)
        return losses * variational_k

    def update_policies(self, batch, network_bundle, act_dict):
        # Unpack predictions from both variational and latent policies
        (variational_k, variational_z, latent_k, latent_z, output_actions, output_actions_uncorrected, output_actions_latent,
         output_actions_latent_uncorrected) = act_dict['variational_k'], act_dict['variational_z'], act_dict['latent_k'], act_dict['latent_z'], \
            act_dict['output_actions'], act_dict['output_actions_uncorrected'], act_dict['output_actions_latent'], act_dict['output_actions_latent_uncorrected']

        # Calculate behavior cloning losses
        b, t, _ = latent_k.shape
        # BC loss including a residual network. Not used by default. 
        weighted_loss = self.calculate_weighted_loss(variational_k.detach(),
                                                       lambda i: self.get_prediction_loss(output_actions[i], batch['actions'], batch['task_length']))
        # BC loss
        weighted_uncorrected_loss = self.calculate_weighted_loss(variational_k,
                                                     lambda i: self.get_prediction_loss(output_actions_uncorrected[i], batch['actions'], batch['task_length']))
        # BC loss using the skill predictions from pi^K and pi^Z
        weighted_latent_loss = self.calculate_weighted_loss(latent_k,
                                                     lambda i: self.get_prediction_loss(output_actions_latent[i], batch['actions'], batch['task_length']))
        # Variance of the GMM action heads
        weighted_variance = self.calculate_weighted_loss(variational_k,
                                                         lambda i: self.get_masked_variance(output_actions[i], batch['task_length']))
        weighted_mse = self.calculate_weighted_loss(variational_k,
                                                    lambda i: self.get_masked_mse_loss(output_actions[i], batch['actions'], batch['task_length']))
        subpolicy_loss = weighted_loss.sum() / sum(batch['task_length'])
        subpolicy_loss_uncorrected  = weighted_uncorrected_loss.sum() / sum(batch['task_length'])
        subpolicy_latent_loss = weighted_latent_loss.sum() / sum(batch['task_length'])
        mse_loss = weighted_mse.sum() / (self.args.action_dim * sum(batch['task_length']))
        avg_variance = weighted_variance.sum() / ((self.args.action_dim - 1) * sum(batch['task_length']))
        # Collect skill usage statistics for logging
        greedy_k = variational_k.argmax(dim = -1)
        greedy_k_latent = latent_k.argmax(dim = -1)
        num_switches, num_skills, correct_k_predictions = 0, 0, 0
        continuous_param = torch.zeros((b, t, self.args.z_dimensions)).to(device)
        for batch_idx in range(b):
            traj_len = batch['task_length'][batch_idx].item()
            traj_k = greedy_k[batch_idx][:traj_len]
            traj_k_latent = greedy_k_latent[batch_idx][:traj_len]
            num_switches += torch.sum(traj_k[1:] != traj_k[:-1]) + 1
            num_skills += torch.unique(traj_k).flatten().shape[0]
            correct_k_predictions += torch.sum(traj_k_latent == traj_k)
            continuous_param[batch_idx, :traj_len] = variational_z[batch_idx][greedy_k[batch_idx, :traj_len]]
        num_switches, num_skills = num_switches / b, num_skills / b
        correct_k_predictions = correct_k_predictions / sum(batch['task_length'])
        center_loss = (continuous_param ** 2).sum() / sum(batch['task_length'])
        # Calculate KL divergence to train latent policy to match variational policy
        variational_k_logprobabilities = torch.log(variational_k + 1e-20)
        latent_k_logprobabilities = torch.log(latent_k + 1e-20)
        target_z = variational_z.reshape(b, 1, self.args.number_policies, self.args.z_dimensions).repeat(1, t, 1, 1)
        if self.args.mask_latent_to_var:
            variational_k_logprobabilities = variational_k_logprobabilities.detach()
            target_z = target_z.detach()
        # KL divergence between pi^K and q
        k_kl_div = torch.nn.functional.kl_div(latent_k_logprobabilities, variational_k_logprobabilities, reduction='none', log_target=True)
        # "KL divergence" between pi^Z and q. Since we use constant variances for continuous parameters, this simplifies to MSE loss. 
        weighted_z_loss = self.mse_loss(latent_z, target_z).sum(dim = -1) * variational_k.detach()
        k_kl_div = mask_tensor(k_kl_div, batch['task_length'])
        weighted_z_loss = mask_tensor(weighted_z_loss, batch['task_length'])
        entropy = mask_tensor(torch.distributions.Categorical(variational_k).entropy(), batch['task_length'])
        entropy = torch.sum(entropy).detach() / sum(batch['task_length'])
        k_kl_loss = torch.sum(k_kl_div) / sum(batch['task_length'])
        z_mse_loss = torch.sum(weighted_z_loss) / (sum(batch['task_length']) * self.args.z_dimensions)
        discriminator_loss = self.update_discriminator_and_get_loss(batch, variational_k, variational_z, network_bundle)
        # Calculate total loss
        total_loss = (subpolicy_loss + subpolicy_loss_uncorrected +
                           self.args.center_loss_weight * center_loss +
                           self.args.kl_loss_weight * k_kl_loss +
                           self.args.z_loss_weight * z_mse_loss +
                           self.args.discriminator_loss_weight * discriminator_loss)
        network_bundle.optimizer.zero_grad()
        total_loss.backward()
        network_bundle.optimizer.step()
        network_bundle.optimizer.zero_grad()
        self.upload_log({"policy loss": torch.mean(subpolicy_loss).cpu().detach().numpy(),
                         "policy loss uncorrected ": torch.mean(subpolicy_loss_uncorrected).cpu().detach().numpy(),
                         "total loss": torch.mean(total_loss).cpu().detach().numpy(),
                         "train mse loss": mse_loss,
                         "discrimiator loss": -discriminator_loss, # the value is flipped here
                         "policy loss latent": torch.mean(subpolicy_latent_loss).cpu().detach().numpy(),
                         "gmm avg variance": avg_variance.detach(),
                         "avg switches per traj": num_switches,
                         "avg skills used per traj": num_skills,
                         "center loss": center_loss,
                         "percent correct k predictions": correct_k_predictions,
                         "k kl loss": k_kl_loss,
                         "z mse loss": z_mse_loss,
                         "variational k entropy": entropy
                         }, step=self.counter)

    def visualize_learned_skills(self, network_bundle):
        z = []
        task_idx = []
        segment_position = []
        for (i, batch) in enumerate(self.task_dataloader):
            with torch.no_grad():
                torch.cuda.empty_cache()
                batch = put_batch_on_device(batch)
                sample_traj, robot_states, old_concatenated_traj = self.collect_inputs(batch, self.network_bundle.image_encoders, self.network_bundle.language_encoder)
                variational_k, variational_z = network_bundle.variational_policy.forward(
                    input=old_concatenated_traj, language_input=batch["task_emb"], task_lengths=batch["task_length"])
                
            # get the most likely discrete skill for each timestep, and the corresponding continuous parameters.
            greedy_k = variational_k.argmax(dim=-1)
            b, t = greedy_k.shape
            continuous_param = torch.zeros((b, t, self.args.z_dimensions)).to(device)
            for batch_idx in range(b):
                continuous_param[batch_idx] = variational_z[batch_idx][greedy_k[batch_idx]]
            if i < 1:
                # Log a video for one of the trajectories.
                visualize_traj_partitioning(batch, greedy_k, continuous_param, self.args.train, self.counter)
            for batch_idx in range(b):
                traj_len = batch['task_length'][batch_idx].item()
                selected_k = greedy_k[batch_idx][:traj_len]
                used_k = torch.unique(selected_k).to(device)
                used_z = variational_z[batch_idx][used_k]
                z.append(used_z)
                segment_position.extend(used_k.cpu().tolist())
                task_idx.extend(batch['task_index'][batch_idx].repeat(used_k.shape[0]).cpu().tolist())
            del batch, sample_traj, old_concatenated_traj
        z = torch.cat(z, dim=0)
        assert z.shape[0] == len(task_idx) == len(segment_position)
        # Store a scatter plot showing the different continuous parameters used as a function of task and discrete skill. 
        store_scatter(z, task_idx, segment_position, self.counter)

    def update_policies_downstream(self, network_bundle, task_bundle, batch, valid_batch, act_dict, act_dict_valid):
        (variational_k, variational_z, latent_k, latent_z, output_actions, output_actions_uncorrected, output_actions_latent,
         output_actions_latent_uncorrected) = act_dict['variational_k'], act_dict['variational_z'], act_dict['latent_k'], act_dict['latent_z'], \
            act_dict['output_actions'], act_dict['output_actions_uncorrected'], act_dict['output_actions_latent'], act_dict['output_actions_latent_uncorrected']
        output_actions_latent_valid, latent_k_valid = act_dict_valid['output_actions_latent'], act_dict_valid['latent_k']
        # calculate reconstruction losses
        b, t, _ = latent_k.shape
        weighted_loss = self.calculate_weighted_loss(latent_k.detach(),
                                                     lambda i: self.get_prediction_loss(output_actions_latent[i], batch['actions'], batch['task_length']))
        weighted_loss_uncorrected = self.calculate_weighted_loss(latent_k,
                                                     lambda i: self.get_prediction_loss(output_actions_latent_uncorrected[i], batch['actions'], batch['task_length']))
        weighted_loss_var = self.calculate_weighted_loss(variational_k,
                                                         lambda i: self.get_prediction_loss(output_actions[i], batch['actions'], batch['task_length']))
        weighted_variance = self.calculate_weighted_loss(latent_k.detach(),
                                                         lambda i: self.get_masked_variance(output_actions_latent[i], batch['task_length']))
        weighted_loss_valid = self.calculate_weighted_loss(latent_k_valid,
                                                           lambda i: self.get_prediction_loss(output_actions_latent_valid[i], valid_batch['actions'], valid_batch['task_length']))
        subpolicy_loss = weighted_loss.sum() / sum(batch['task_length'])
        subpolicy_loss_uncorrected = weighted_loss_uncorrected.sum() / sum(batch['task_length'])
        subpolicy_loss_valid = weighted_loss_valid.sum() / sum(valid_batch['task_length'])
        subpolicy_loss_var = weighted_loss_var.sum() / sum(batch['task_length'])
        avg_variance = weighted_variance.sum() / ((self.args.action_dim - 1) * sum(batch['task_length']))

        # calculate gap between latent and variational skill predictions
        variational_k_logprobabilities = torch.log(variational_k + 1e-20)
        latent_k_logprobabilities = torch.log(latent_k + 1e-20)
        k_kl_div = torch.nn.functional.kl_div(latent_k_logprobabilities, variational_k_logprobabilities, reduction='none', log_target=True)
        k_kl_div = mask_tensor(k_kl_div, batch['task_length'])
        k_kl_loss = torch.sum(k_kl_div) / sum(batch['task_length'])
        target_z = variational_z.reshape(b, 1, self.args.number_policies, self.args.z_dimensions).repeat(1, self.args.traj_length, 1, 1)
        z_loss = self.mse_loss(latent_z, target_z)
        weighted_z_loss = z_loss.sum(dim=-1) * variational_k.detach()
        weighted_z_loss = mask_tensor(weighted_z_loss, batch['task_length'])
        z_mse_loss = torch.sum(weighted_z_loss) / (sum(batch['task_length']) * self.args.z_dimensions)
        total_loss = subpolicy_loss + subpolicy_loss_uncorrected + \
                     self.args.kl_loss_weight * k_kl_loss + \
                     self.args.z_loss_weight * z_mse_loss + \
                     -1 * self.args.variance_reg_weight * avg_variance

        network_bundle.optimizer.zero_grad()
        total_loss.backward()
        network_bundle.optimizer.step()
        Logger.log({f"Task {task_bundle.task_idx} logp loss": subpolicy_loss,
                   f"Task {task_bundle.task_idx} logp loss uncorrected": subpolicy_loss_uncorrected,
                   f"Task {task_bundle.task_idx} logp loss validation": subpolicy_loss_valid,
                   f"Task {task_bundle.task_idx} validation loss ratio": subpolicy_loss_valid / subpolicy_loss,
                   f"Task {task_bundle.task_idx} logp loss variational": subpolicy_loss_var,
                   f"Task {task_bundle.task_idx} k kl loss": k_kl_loss,
                   f"Task {task_bundle.task_idx} z mse loss": z_mse_loss,
                   f"Task {task_bundle.task_idx} total loss": total_loss,
                   f"Task {task_bundle.task_idx} counter": task_bundle.counter,
                   f"Task {task_bundle.task_idx} avg variance": avg_variance.detach(),
                   })
        
    # Get action during environment rollout using latent (online) policy
    @torch.no_grad()
    def get_rollout_action(self, network_bundle, sample_traj, robot_state, lang_embedding):
        # Use latent policy for online execution (no access to future/actions)
        latent_k = network_bundle.discrete_policy.forward(input=sample_traj, language=lang_embedding)[0]
        latent_z = network_bundle.continuous_policy.forward(input=sample_traj, language=lang_embedding)[0]
        greedy_k = torch.argmax(latent_k, dim=-1)
        k = greedy_k[-1].item()
        t = latent_z.shape[0]
        filtered_z = latent_z[torch.arange(t).to(device), greedy_k].unsqueeze(dim=0)
        if self.args.use_corrected_action:
            action_idx = 0
        else:
            action_idx = 1
        selected_a = network_bundle.primitive_policy_set[0].get_actions(sample_traj, robot_state, filtered_z, greedy_k)[action_idx]
        return torch.flatten(selected_a[:, -1]).unsqueeze(dim = 0).cpu().detach().numpy()



