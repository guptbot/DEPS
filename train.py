import torch
from utils.utils import *
from utils.visualization_tools import *

from networks.networks import (VariationalNetwork, DiscretePolicyNetwork, ContinuousPolicyNetwork, StochasticLowLevelPolicyNetwork,
                      TaskPredictionMLP, IMAGE_ENCODER, StochasticLowLevelPolicyNetworkWithCompression, ParaskillStdPolicyNetwork, IMG_AUG)
from env.metaworld_spatial_encode import spatial_encode
import time
from utils.logger import Logger
from env.metaworld_rollout import MetaworldRollout
import cv2
import sys, os
import numpy as np
import copy
from mw.metaworld_tasks import TASK_IDX_TO_TASK_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Container for all neural networks used in training
class NetworkBundle(torch.nn.Module):
    def __init__(self, variational_policy, discrete_policy, continuous_policy, primitive_policy_set, discriminator, image_encoder, optimizer = None, discriminator_optimizer = None):
        super().__init__()
        self.variational_policy = variational_policy  # q
        self.discrete_policy = discrete_policy  # pi^K
        self.continuous_policy = continuous_policy  # pi^Z
        self.image_encoder = image_encoder
        self.discriminator = discriminator  # Not used
        self.primitive_policy_set = torch.nn.ModuleList(primitive_policy_set)  # Set of policies pi^A
        #self.language_encoder = language_encoder
        self.optimizer = optimizer
        self.discriminator_optimizer = discriminator_optimizer  # Not used
        return


# Bundle of task-specific data for fine-tuning on a single downstream task
class FineTuneTaskBundle():
    def __init__(self, train_dataloader, task_idx):
        self.train_dataloader = train_dataloader
        #self.validation_dataloader_generator = valid_dataloader_generator
        self.task_idx = task_idx
        self.counter = 0
        self.epoch = 0
        self.rollout_traj_index = 0
        return

# Base class for training policies - implements common training logic
# Subclasses (deps, depsUncompressed, Multitask) implement algorithm-specific methods
class PolicyManagerBase():
    def __init__(self, dataloader=None, test_dataloader = None, task_dataloader = None, args=None):
        super(PolicyManagerBase, self).__init__()

        self.args = args
        self.dataloader = dataloader  # the training dataloader
        #self.valid_dataloader_generator = dataloader_generator(valid_dataloader)
        self.test_dataloader_generator = dataloader_generator(test_dataloader)
        self.task_dataloader = task_dataloader
        self.setup()

    # Methods implemented by algorithm-specific subclasses #############
    def get_actions(self, batch, network_bundle, sample_traj, robot_states, old_concatenated_traj):
        raise NotImplementedError

    def update_policies(self, batch, network_bundle, act_dict):
        raise NotImplementedError

    def visualize_learned_skills(self, network_bundle):
        raise NotImplementedError

    def calculate_cloning_loss(self, batch, network_bundle, sample_traj, robot_states, old_concatenated_traj):
        raise NotImplementedError

    def update_policies_downstream(self, network_bundle, task_bundle, batch, act_dict):
        raise NotImplementedError

    def get_rollout_action(self, network_bundle, sample_traj, robot_state, task_index):
        raise NotImplementedError

    def get_one_hot_task_tensor(self, batch):
        task_index = batch['task_index']
        return torch.nn.functional.one_hot(task_index, num_classes=50)

    # Train discriminator to predict task from skill embeddings (adversarial regularization)
    # NOTE: this is not used in the final version
    def update_discriminator_and_get_loss(self, batch, variational_k, variational_z, network_bundle, valid = False):
        skill_probs = torch.distributions.Categorical(probs=variational_k)
        sampled_k = skill_probs.sample()
        inp = []
        target = []
        for batch_idx in range(sampled_k.shape[0]):
            task_length = batch['task_length'][batch_idx].item()
            k = sampled_k[batch_idx][:task_length]
            inp.append(variational_z[batch_idx][k])
            target.append(torch.full((task_length,), batch['task_index'][batch_idx].item()).to(device))
        inp = torch.cat(inp, dim = 0)
        target = torch.cat(target, dim = 0)
        # first update the discriminator (if valid is False)
        network_bundle.discriminator_optimizer.zero_grad()
        if not valid:
            out = network_bundle.discriminator.forward(inp.detach())
            loss = torch.nn.CrossEntropyLoss()(out, target)
            loss.backward()
            network_bundle.discriminator_optimizer.step()
        # then get the same output again
        network_bundle.discriminator_optimizer.zero_grad()
        out = network_bundle.discriminator.forward(inp)
        loss = torch.nn.CrossEntropyLoss()(out, target)
        return -loss # we want to MAXIMIZE this loss

    def setup(self):
        # Init WandB
        if self.args.debug:
            Logger.init(project=None, config=None, dir=None, name=None, mode="disabled")
        elif self.args.finetune:
            Logger.init(
                project="parameterized-skill-mw-small", config=self.args, dir="../scratch/wandb",
                name=self.args.name, entity="parameterized-skills-2")
        else:
            Logger.init(
                project="parameterized-skill-mw-small", config=self.args, dir="../scratch/wandb",
                name=self.args.name, entity="parameterized-skills-2")
        np.random.seed(seed=self.args.seed)
        torch.manual_seed(self.args.seed)
        np.set_printoptions(suppress=True, precision=4)
        # Initialize networks
        self.create_networks()
        # Create loss functions and initialize optimizers
        self.create_training_ops()
        logdir = os.path.join(self.args.logdir, self.args.name)
        if not (os.path.isdir(logdir)):
            os.mkdir(logdir)
        self.savedir = os.path.join(logdir, "saved_models")
        if not (os.path.isdir(self.savedir)):
            os.mkdir(self.savedir)

    def create_networks(self):
        # Determine input size for primitive policies (image features or robot state)
        if self.args.image_to_subpolicy:
            primitive_input_sz = self.args.state_dim
        else:
            primitive_input_sz = self.args.robot_state_dim
        primitive_policy_set = []  # Set of low-level skill policies

        for i in range(1):  # NOTE: we use a single low level policy for all discrete skills (i.e. DO NOT use separate networks for different discrete skills.)
            if self.args.algo == "deps":
                primitive_policy = StochasticLowLevelPolicyNetworkWithCompression(args = self.args).to(device)
            elif self.args.algo == 'deps_uncompressed':
                primitive_policy = ParaskillStdPolicyNetwork(args=self.args, input_size=primitive_input_sz).to(device)
            elif self.args.algo == 'multitask':
                primitive_policy = StochasticLowLevelPolicyNetwork(args=self.args, input_size=primitive_input_sz).to(device)
            else:
                raise Exception("this should not happen, unknown algo")
            primitive_policy_set.append(primitive_policy)
        # Variational network (q)
        variational_policy = VariationalNetwork(args = self.args).to(device)
        # Discrete policy: selects skill index autoregressively (pi^K)
        discrete_policy = DiscretePolicyNetwork(args = self.args).to(device)
        # Continuous policy: generates continuous skill parameters autoregressively (pi^Z)
        continuous_policy = ContinuousPolicyNetwork(args = self.args).to(device)
        discriminator = TaskPredictionMLP(args = self.args).to(device)  # Not used
        self.network_bundle = NetworkBundle(variational_policy, discrete_policy, continuous_policy, primitive_policy_set, discriminator, IMAGE_ENCODER)
        self.network_bundle.train()

    def create_training_ops(self):
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        # Collect all trainable parameters from all networks
        parameter_list = list(self.network_bundle.discrete_policy.parameters()) + list(self.network_bundle.variational_policy.parameters()) \
                         + list(self.network_bundle.continuous_policy.parameters())
        for primitive_policy in self.network_bundle.primitive_policy_set:
            parameter_list = parameter_list + list(primitive_policy.parameters())
        parameter_list = parameter_list + list(self.network_bundle.image_encoder.parameters())
        # Initialize the optimizer with parameters of all the models
        self.network_bundle.optimizer = torch.optim.Adam(parameter_list, lr=self.args.learning_rate)
        discriminator_parameter_list = list(self.network_bundle.discriminator.parameters())
        self.network_bundle.discriminator_optimizer = torch.optim.Adam(discriminator_parameter_list, lr=self.args.learning_rate)

    def create_optimizer(self, continuous_policy, discrete_policy, sub_policy, image_encoder):
        """Create optimizer for finetuning on a single task"""
        parameter_list = list(discrete_policy.parameters()) + list(continuous_policy.parameters())
        if self.args.finetune_resnet:  # Also finetune the resnet encoder
            parameter_list = parameter_list + list(image_encoder.parameters())
        if self.args.finetune_subpolicy:  # Also finetune the low level policy
            for idx in range(len(sub_policy)):
                parameter_list = parameter_list + list(sub_policy[idx].parameters())
        return torch.optim.Adam(parameter_list, lr=self.args.learning_rate)

    def upload_log(self, mylog, step):
        Logger.log(mylog, step=step)

    def load_all_models(self, path, just_subpolicy=False):
        load_object = torch.load(path)
        for i in range(len(self.network_bundle.primitive_policy_set)):
            self.network_bundle.primitive_policy_set[i].load_state_dict(load_object['Primitive_Policy' + str(i)])
        self.network_bundle.discrete_policy.load_state_dict(load_object['Discrete_Policy'])
        self.network_bundle.continuous_policy.load_state_dict(load_object['Continuous_Policy'])
        self.network_bundle.variational_policy.load_state_dict(load_object['Variational_Policy'])
        self.network_bundle.image_encoder.load_state_dict(load_object['Image_encoder'])
        self.network_bundle.discriminator.load_state_dict(load_object['Discriminator'])

    def save_all_models(self, suffix):
        save_object = {}
        save_object['Variational_Policy'] = self.network_bundle.variational_policy.state_dict()
        save_object['Discrete_Policy'] = self.network_bundle.discrete_policy.state_dict()
        save_object['Continuous_Policy'] = self.network_bundle.continuous_policy.state_dict()
        for i in range(len(self.network_bundle.primitive_policy_set)):
            save_object['Primitive_Policy' + str(i)] = self.network_bundle.primitive_policy_set[i].state_dict()
        save_object['Image_encoder'] = self.network_bundle.image_encoder.state_dict()
        save_object['Discriminator'] = self.network_bundle.discriminator.state_dict()

        torch.save(save_object, os.path.join(self.savedir, "Model_" + suffix))

    def get_prediction_loss(self, prediction, true_actions, task_lengths):
        # log_probs = prediction.log_prob(true_actions)
        # log_probs = mask_tensor(log_probs, task_lengths)
        # loss = -log_probs
        loss = torch.nn.functional.l1_loss(prediction, true_actions.float(), reduction = 'none')
        loss = mask_tensor(loss, task_lengths)
        loss = torch.sum(loss, dim = -1)
        return loss

    def get_masked_variance(self, actions, task_lengths):
        variance = torch.mean(actions.component_distribution.base_dist.scale, dim = 2)
        variance = mask_tensor(variance, task_lengths)
        variances = variance.sum(dim=-1)
        return variances

    def get_masked_mse_loss(self, actions, true_actions, task_lengths):  # NOTE: only for logging, NOT used in gradient calculation
        sampled_actions = actions.sample().detach()
        masked_sampled_actions = true_actions.clone().detach()
        for batch_idx in range(sampled_actions.shape[0]):
            traj_len = task_lengths[batch_idx].item()
            masked_sampled_actions[batch_idx, :traj_len] = sampled_actions[batch_idx, :traj_len]
        mse_loss = self.mse_loss(masked_sampled_actions, true_actions).sum(dim = -1)
        return mse_loss

    # Main function to do all training
    def train(self):
        if self.args.model:
            self.load_all_models(self.args.model)
        self.counter = 0
        self.visualize_learned_skills(self.network_bundle)
        for e in range(self.args.epochs):
            print(f"Starting Epoch: {e}")
            if e != 0 and not self.args.debug: self.save_all_models(f"epoch{e}")
            for (i, batch) in enumerate(self.dataloader):
                print(f"Epoch: {e}, Step: {i}")
                torch.cuda.empty_cache()
                batch = put_batch_on_device(batch)
                if self.args.apply_aug:
                    batch = self.apply_augmentation(batch)
                sample_traj, robot_states, old_concatenated_traj = self.collect_inputs(batch, self.network_bundle.image_encoder)
                #sample_traj = sample_traj.detach()
                act_dict = self.get_actions(batch, self.network_bundle, sample_traj, robot_states, old_concatenated_traj)
                self.update_policies(batch, self.network_bundle, act_dict)
                #self.automatic_evaluation(self.valid_dataloader_generator, "validation")
                if self.counter % 10 == 0:
                    self.automatic_evaluation(self.test_dataloader_generator, "test")
                self.counter += 1
                if self.args.debug: break  # to speed up debugging process
            self.visualize_learned_skills(self.network_bundle)

    def get_img_tuple(self, data):
        img_tuple = tuple(
            [data["imgs"]]
        )
        return img_tuple
    # def get_aug_output_dict(self, out):
    #     img_dict = {
    #         img_name: out[idx]
    #         for idx, img_name in enumerate(self.network_bundle.image_encoders.keys())
    #     }
    #     return img_dict
    def apply_augmentation(self, batch):  # Apply data augmentation (translation) to images
        img_tuple = self.get_img_tuple(batch)
        aug_out = IMG_AUG(img_tuple)
        assert len(aug_out) == 1
        batch["imgs"] =  aug_out[0]
        return batch

    # Encode observations (images + robot state) into feature vectors
    def collect_inputs(self, batch, image_encoder):
        sample_trajs, robot_states = spatial_encode(batch, use_language=self.args.use_language, image_encoder = image_encoder)
        old_concatenated_trajs = torch.cat([sample_trajs, batch['actions']], dim=-1)  # Concatenate with actions for variational inference
        return sample_trajs, robot_states, old_concatenated_trajs

    def automatic_evaluation(self, dataloader, label = None):
        """Evaluate the BC loss of the learned policy on validation/test trajectories"""
        self.args.add_noise_to_z = 0
        batch = next(dataloader)
        with torch.no_grad():
            batch = put_batch_on_device(batch)
            sample_traj, robot_states, old_concatenated_traj = self.collect_inputs(batch, self.network_bundle.image_encoder)
            b, t, s = sample_traj.shape
            assert b == 1  # is used below for indexing
            loss = self.calculate_cloning_loss(batch, self.network_bundle, sample_traj, robot_states, old_concatenated_traj)
        self.args.add_noise_to_z = 1
        self.upload_log({f"{label} loss": loss}, step=self.counter)

    # Fine-tune pretrained model on downstream tasks
    def finetune(self, num_tasks, train_dataloaders):
        if self.args.finetune_resnet:
            if self.args.kl_loss_weight != 0 or self.args.z_loss_weight != 0:
                raise Exception("cant use variational network as prior while also finetuning resnet")
        if self.args.model:
            self.load_all_models(self.args.model)
        self.network_bundle.to("cpu")

        # Remove the DataParallel wrapper from the image encoders
        # for key in self.network_bundle.image_encoders:
        #     self.network_bundle.image_encoders[key] = self.network_bundle.image_encoders[key].module
        # Initialize all relevant networks for each task
        average_success_rates = [0 for _ in range(1, self.args.epochs + 1, self.args.downstream_rollout_frequ)]
        # Make a separate copy of the networks for each task. Each task is finetuned separately.
        for task_idx in range(num_tasks):
            variational_policy = copy.deepcopy(self.network_bundle.variational_policy).to(device)
            continuous_policy = copy.deepcopy(self.network_bundle.continuous_policy).to(device)
            discrete_policy = copy.deepcopy(self.network_bundle.discrete_policy).to(device)
            primitive_policy = []
            for idx in range(len(self.network_bundle.primitive_policy_set)):
                primitive_policy.append(copy.deepcopy(self.network_bundle.primitive_policy_set[idx]).to(device))
            if self.args.finetune_resnet:
                image_encoder = copy.deepcopy(self.network_bundle.image_encoder).to(device)
                # for key in image_encoders:
                #     image_encoders[key] = torch.nn.DataParallel(image_encoders[key].to(device))
            else:
                image_encoder = self.network_bundle.image_encoder.to(device)
            optimizer = self.create_optimizer(continuous_policy=continuous_policy,
                                              discrete_policy=discrete_policy,
                                              sub_policy=primitive_policy,
                                              image_encoder = image_encoder)
            network_bundle = NetworkBundle(variational_policy, discrete_policy, continuous_policy, primitive_policy, None, image_encoder, optimizer)
            network_bundle.train()
            sample_batch = next(iter(train_dataloaders[task_idx]))
            real_task_idx = self.args.test_tasks[task_idx]
            assert real_task_idx == torch.flatten(sample_batch['task_index'])[0].item()
            #valid_sample_batch = next(iter(valid_dataloaders[task_idx]))
            #assert False not in (sample_batch['task_emb'][0] == valid_sample_batch['task_emb'][0])
            task_bundle = FineTuneTaskBundle(
                train_dataloader = train_dataloaders[task_idx],
                #valid_dataloader_generator = dataloader_generator(valid_dataloaders[task_idx]),
                task_idx = real_task_idx)


            # Init axes for wandb plotting
            Logger.define_metric(f"Task {task_idx} epoch", hidden = True)
            Logger.define_metric(f"Task {task_idx} counter", hidden=True)
            Logger.define_metric(f"Task {task_idx} test success", step_metric = f"Task {task_idx} epoch")
            Logger.define_metric(f"AVERAGE SUCCESS RATE", step_metric=f"Task {task_idx} epoch")
            Logger.define_metric(f"Task {task_bundle.task_idx} logp loss", step_metric = f"Task {task_idx} counter")
            #Logger.define_metric(f"Task {task_bundle.task_idx} logp loss validation", step_metric = f"Task {task_idx} counter")
            #Logger.define_metric(f"Task {task_bundle.task_idx} validation loss ratio", step_metric = f"Task {task_idx} counter")
            Logger.define_metric(f"Task {task_bundle.task_idx} logp loss variational", step_metric=f"Task {task_idx} counter")
            Logger.define_metric(f"Task {task_bundle.task_idx} k kl loss", step_metric = f"Task {task_idx} counter")
            Logger.define_metric(f"Task {task_bundle.task_idx} z mse loss", step_metric = f"Task {task_idx} counter")
            Logger.define_metric(f"Task {task_bundle.task_idx} total loss", step_metric = f"Task {task_idx} counter")
            Logger.define_metric(f"Task {task_bundle.task_idx} avg variance", step_metric = f"Task {task_idx} counter")

            epoch = 0
            self.success_rates = []
            for _ in range(self.args.epochs):
                epoch += 1
                print("STARTING NEW EPOCH")
                print(f"Task: {task_idx}")
                self.run_epoch_singletask(task_bundle, network_bundle)
                if epoch % self.args.downstream_rollout_frequ == 0:
                    self.eval_finetuning(task_bundle, network_bundle)
            average_success_rates = [average_success_rates[i] + self.success_rates[i] for i in range(len(average_success_rates))]
            network_bundle.to("cpu")
            torch.cuda.empty_cache()
        Logger.define_metric("EPOCH", hidden=True)
        Logger.define_metric("AVERAGE SUCCESS RATE", step_metric="EPOCH")
        for i in range(len(average_success_rates)):
            Logger.log({"AVERAGE SUCCESS RATE": average_success_rates[i] / num_tasks, "EPOCH": (i + 1) * self.args.downstream_rollout_frequ})

    def run_epoch_singletask(self, task_bundle, network_bundle):
        """Run a single epoch of finetuning on a single task"""
        task_bundle.epoch += 1
        for batch in task_bundle.train_dataloader:
            print("STARTING NEW ITERATION")
            #validation_batch = next(task_bundle.validation_dataloader_generator)
            task_bundle.counter += 1
            batch = put_batch_on_device(batch)
            sample_traj, robot_states, old_concatenated_traj = self.collect_inputs(batch, network_bundle.image_encoder)
            action_dict = self.get_actions(batch, network_bundle, sample_traj, robot_states, old_concatenated_traj)

            # with torch.no_grad():
            #     validation_batch = put_batch_on_device(validation_batch)
            #     validation_traj, validation_robot_states, old_concatenated_traj_valid = self.collect_inputs(validation_batch, network_bundle.image_encoders, network_bundle.language_encoder)
            #     action_dict_valid = self.get_actions(validation_batch, network_bundle, validation_traj, validation_robot_states, old_concatenated_traj_valid)

            self.update_policies_downstream(network_bundle, task_bundle, batch, action_dict)
            del batch, sample_traj, robot_states, old_concatenated_traj
            torch.cuda.empty_cache()
            if self.args.debug: break


    def eval_finetuning(self, task_bundle, network_bundle):
        NUM_TESTS = 20
        num_successes = 0
        for i in range(NUM_TESTS):
            num_successes += self.rollout_single_network(task_bundle, network_bundle, i == 0)
        Logger.log({f"Task {task_bundle.task_idx} test success": num_successes / NUM_TESTS,
                   f"Task {task_bundle.task_idx} epoch": task_bundle.epoch})
        self.success_rates.append(num_successes / NUM_TESTS)

    # Execute a single rollout in the environment to test the policy
    @torch.no_grad()
    def rollout_single_network(self, task_bundle, network_bundle, save_vid = False):
        network_bundle.eval()
        num_iter = 600  # Maximum rollout length
        rollout_trajectories = list(range(self.args.rollout_trajectories_start_idx, self.args.rollout_trajectories_end_idx))
        traj_index = rollout_trajectories[task_bundle.rollout_traj_index % len(rollout_trajectories)]
        task_bundle.rollout_traj_index += 1
        task_idx = task_bundle.task_idx
        task_name = TASK_IDX_TO_TASK_NAME[task_idx]
        one_hot_task_idx = torch.nn.functional.one_hot(torch.tensor(task_idx).to(device), num_classes=50).float().reshape(1, -1)

        env = MetaworldRollout(task_name, traj_index)
        obs_dict = env.get_data()
        obs, robot_state = spatial_encode(obs_dict, self.args.use_language, image_encoder = network_bundle.image_encoder)
        obs = obs.detach()
        sample_traj = obs
        images = []

        for t in range(num_iter - 1):
            action_to_execute = self.get_rollout_action(network_bundle, sample_traj, robot_state, one_hot_task_idx)
            obs_dict, reward, terminal, truncated, info = env.step(action_to_execute)
            images.append(obs_dict['imgs'][0][0])
            new_state, new_robot_state = spatial_encode(obs_dict, use_language=self.args.use_language, image_encoder = network_bundle.image_encoder)
            new_state = new_state.detach()
            sample_traj = torch.cat((sample_traj, new_state), dim = 1)
            robot_state = torch.cat((robot_state, new_robot_state), dim = 1)
            if info['success'] == 1.0:
                success = 1
                break
            elif terminal or truncated:
                print("THE ROLLOUT FAILED:(")
                success = 0
                break
        else:
            print("THE ROLLOUT FAILED:(")
            success = 0
        if save_vid:
            save_video(images, task_bundle.task_idx, task_bundle.epoch)
        env.terminate()
        del images, robot_state, sample_traj
        network_bundle.train()
        return success



