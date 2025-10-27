from torch.nn import functional as F
from utils.utils import *
import torch
import torch.nn as nn
from networks.resnet import DataAugGroup, TranslationAug, Encoder
from networks.gmm_head import GMMHead


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Variational network: bidirectional GRU that infers skills from full trajectory (q)
class VariationalNetwork(torch.nn.Module):
    def __init__(self, args):
        super(VariationalNetwork, self).__init__()
        self.args = args
        # Layers
        #assert args.number_policies != 1
        self.lstm = torch.nn.GRU(input_size=args.input_size + args.language_dim, hidden_size=args.hidden_size,
                                 num_layers=args.variational_latent_num_layers, bidirectional=True, batch_first=True)
        self.k_first_layer = torch.nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.k_final_layer = torch.nn.Linear(args.hidden_size, args.number_policies)
        self.z_first_layer = torch.nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.z_final_layer = torch.nn.Linear(args.hidden_size, args.number_policies * args.z_dimensions)
        # Activation funcs
        self.activation_layer = torch.nn.Tanh()
        self.hidden_activation = F.relu

    def forward(self, input, task_indices, task_lengths):
        # Append task encoding to visual input (DEPS_mw uses one-hot task encoding)
        task_indices = task_indices.unsqueeze(1).repeat(1, input.shape[1], 1)
        input = torch.cat((input, task_indices), dim=-1)

        # Make a packed sequence before passing to lstm. This makes sure the LSTM ignores zeroed-out entries beyond the end of the trajectory.
        input = torch.nn.utils.rnn.pack_padded_sequence(input, list(task_lengths), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.lstm(input)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=self.args.traj_length)
        b, s, _ = outputs.size()
        # Get discrete skill predictions for each timestep
        variational_k = self.hidden_activation(self.k_first_layer(outputs))
        variational_k = self.k_final_layer(variational_k)
        variational_k = torch.nn.functional.softmax(variational_k, dim=-1)
        #variational_k = torch.nn.functional.gumbel_softmax(variational_k, dim=-1, hard = True)

        # Get continuous parameter predictions for each discrete skill
        variational_z = self.hidden_activation(self.z_first_layer(outputs))
        variational_z = self.z_final_layer(variational_z)
        if self.args.variational_z_mean_nonlinearity:  # restrict range to [-1, 1]
            variational_z = self.activation_layer(variational_z)
        variational_z = variational_z.view(b, s, -1)
        variational_z = mask_tensor(variational_z, task_lengths)  # Mask extra elements to 0 so they dont affect averages
        variational_z = variational_z.sum(dim = 1)
        variational_z = variational_z / task_lengths.view(-1, 1)

        return variational_k, variational_z.reshape(-1, self.args.number_policies, self.args.z_dimensions)


# Discrete policy: autoregressively selects skill index at each timestep (pi^K)
class DiscretePolicyNetwork(torch.nn.Module):
    def __init__(self, args):
        super(DiscretePolicyNetwork, self).__init__()
        self.args = args
        # Layers
        self.lstm = torch.nn.GRU(input_size= args.state_dim, hidden_size=args.hidden_size,
                                     num_layers=args.variational_latent_num_layers, batch_first=True, bidirectional = False).to(device)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param, gain=5)
        self.subpolicy_output_layer1 = torch.nn.Linear(args.hidden_size + args.language_dim, args.hidden_size)
        self.subpolicy_output_layer2 = torch.nn.Linear(args.hidden_size, args.number_policies)
        # Activation Funcs
        self.hidden_activation = F.relu

    def forward(self, input, task_indices):
        b, s, _ = input.size()
        task_indices = task_indices.reshape(b, 1, -1).repeat(1, s, 1)
        #concat_input_lang = torch.cat((input, language), dim=-1)
        outputs, _ = self.lstm(input)
        concat_output = torch.cat((outputs, task_indices), dim=-1)
        variational_k = self.hidden_activation(self.subpolicy_output_layer1(concat_output))
        variational_k = self.subpolicy_output_layer2(variational_k)
        variational_k = torch.nn.functional.softmax(variational_k, dim=-1)
        return variational_k

# Continuous policy: autoregressively generates continuous skill parameters at each timestep (pi^Z)
class ContinuousPolicyNetwork(torch.nn.Module):
    def __init__(self, args):
        super(ContinuousPolicyNetwork, self).__init__()
        self.args = args
        # Layers
        self.lstm = torch.nn.GRU(input_size=args.state_dim, hidden_size=args.hidden_size,
                                 num_layers=args.variational_latent_num_layers, batch_first=True, bidirectional = False).to(device)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_normal_(param, gain=5)
        self.subpolicy_output_layer1 = torch.nn.Linear(args.hidden_size + args.language_dim, args.hidden_size)
        self.subpolicy_output_layer2 = torch.nn.Linear(args.hidden_size, args.z_dimensions * args.number_policies)
        # Activation funcs
        self.hidden_activation = F.relu
        self.activation_layer = torch.nn.Tanh()

    def forward(self, input, task_indices):
        b, s, _ = input.size()
        task_indices = task_indices.reshape(b, 1, -1).repeat(1, s, 1)
        #concat_input_lang = torch.cat((input, language), dim=-1)
        outputs, _ = self.lstm(input)
        concat_output = torch.cat((outputs, task_indices), dim=-1)
        variational_z = self.hidden_activation(self.subpolicy_output_layer1(concat_output))
        variational_z = self.subpolicy_output_layer2(variational_z)
        if self.args.variational_z_mean_nonlinearity:
            variational_z = self.activation_layer(variational_z)
        # Return a continuous parameterization for each (timestep, discrete skill). This is indexed into using the selected discrete skill.
        return variational_z.reshape(b, s, self.args.number_policies, self.args.z_dimensions)

# Low-level policy: outputs action distribution conditioned on state and skill parameters
class StochasticLowLevelPolicyNetwork(torch.nn.Module):
    def __init__(self, args, input_size):
        super(StochasticLowLevelPolicyNetwork, self).__init__()
        self.args = args
        # Layers
        self.continuous_param_layer = torch.nn.Linear(args.z_dimensions, int(args.hidden_size_primitive / 2))
        self.embedding_layer = torch.nn.Embedding(self.args.number_policies, int(args.hidden_size_primitive / 2))
        additional_dim = 0
        if args.algo == "multitask":
            additional_dim = args.language_dim
        self.lstm = torch.nn.LSTM(input_size=input_size + additional_dim, hidden_size=args.hidden_size_primitive,
                                  num_layers=args.subpolicy_num_layers)
        self.markov_layer = torch.nn.Linear(input_size + additional_dim, args.hidden_size_primitive)
        self.pre_GMM_layer = torch.nn.Linear(int(args.hidden_size_primitive * 1.5), args.gmm_hidden_size)
        # Initialize GMM head with appropriate settings
        # network_kwargs = {'hidden_size': args.gmm_hidden_size, 'num_layers': args.gmm_num_layers, 'min_std': 0.0001,
        #                   'num_modes': args.num_gaussians, 'low_eval_noise': False, 'activation': 'softplus',
        #                   'input_size': int(args.hidden_size_primitive * 1.5), 'output_size': args.action_dim}
        # loss_kwargs = {'loss_coef': 1.0}
        # self.GMM_head = GMMHead(**network_kwargs, **loss_kwargs)

        self.decoder = nn.Sequential(
            nn.Linear(int(args.hidden_size_primitive * 1.5), args.hidden_size_primitive),
            nn.ReLU(),
            nn.Linear(args.hidden_size_primitive, args.hidden_size_primitive),
            nn.ReLU(),
            nn.Linear(args.hidden_size_primitive, args.action_dim)
        )
        # Activation funcs
        self.hidden_activation = torch.nn.ReLU()

    def forward(self, input, z_vals, skill_idx, task_indices = None):
        b, s, _ = input.size()
        if task_indices is not None:
            task_indices = task_indices.reshape(b, 1, -1).repeat(1, s, 1)
            input = torch.cat((input, task_indices), dim = -1)
        if self.args.markov_subpolicy:
            lstm_outputs = self.hidden_activation(self.markov_layer(input))
        else:
            lstm_outputs, hidden = self.lstm(input)
        z_input = self.continuous_param_layer(z_vals)
        #skill_embedding = self.embedding_layer(skill_idx).reshape(b, s, int(self.args.hidden_size_primitive / 2))
        lstm_outputs_with_z = torch.cat((lstm_outputs, z_input), dim=-1)
        #pre_gmm = self.hidden_activation(self.pre_GMM_layer(lstm_outputs_with_z))
        #dist = self.GMM_head(lstm_outputs_with_z)
        dist = self.decoder(lstm_outputs_with_z)
        return dist

    def get_actions(self, input, z_vals, skill_idx, task_indices = None):
        with torch.no_grad():
            dist = self.forward(input, z_vals, skill_idx, task_indices)
            action = dist.detach()
        return action


# Returns a GMM of actions for each timestep given state and continuous parameters
class ParaskillStdPolicyNetwork(torch.nn.Module):
    def __init__(self, args, input_size):
        super(ParaskillStdPolicyNetwork, self).__init__()
        self.args = args
        # Layers
        self.continuous_param_layer = torch.nn.Linear(args.z_dimensions, int(args.hidden_size_primitive / 2))
        self.embedding_layer = torch.nn.Embedding(self.args.number_policies, int(args.hidden_size_primitive / 2))
        additional_dim = 0
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=args.hidden_size_primitive,
                                  num_layers=args.subpolicy_num_layers)
        self.markov_layer = torch.nn.Linear(input_size + additional_dim, args.hidden_size_primitive)
        self.pre_GMM_layer = torch.nn.Linear(int(args.hidden_size_primitive * 1.5), args.gmm_hidden_size)
        # Initialize GMM head with appropriate settings
        # network_kwargs = {'hidden_size': args.gmm_hidden_size, 'num_layers': args.gmm_num_layers, 'min_std': 0.0001,
        #                   'num_modes': args.num_gaussians, 'low_eval_noise': False, 'activation': 'softplus',
        #                   'input_size': int(args.hidden_size_primitive * 1.5), 'output_size': args.action_dim}
        # loss_kwargs = {'loss_coef': 1.0}
        # self.GMM_head = GMMHead(**network_kwargs, **loss_kwargs)

        self.decoder = nn.Sequential(
            nn.Linear(int(args.hidden_size_primitive * 1.5), args.hidden_size_primitive),
            nn.ReLU(),
            nn.Linear(args.hidden_size_primitive, args.hidden_size_primitive),
            nn.ReLU(),
            nn.Linear(args.hidden_size_primitive, args.action_dim)
        )
        # Activation funcs
        self.hidden_activation = torch.nn.ReLU()

    def forward(self, input, z_vals, skill_idx):
        b, s, _ = input.size()
        if self.args.markov_subpolicy:
            lstm_outputs = self.hidden_activation(self.markov_layer(input))
        else:
            lstm_outputs, hidden = self.lstm(input)
        z_input = self.continuous_param_layer(z_vals)
        #skill_embedding = self.embedding_layer(skill_idx).reshape(b, s, int(self.args.hidden_size_primitive / 2))
        lstm_outputs_with_z = torch.cat((lstm_outputs, z_input), dim=-1)
        #pre_gmm = self.hidden_activation(self.pre_GMM_layer(lstm_outputs_with_z))
        #dist = self.GMM_head(lstm_outputs_with_z)
        dist = self.decoder(lstm_outputs_with_z)
        return dist

    def get_actions(self, input, z_vals, skill_idx):
        with torch.no_grad():
            dist = self.forward(input, z_vals, skill_idx)
            action = dist.detach()
        return action


# Low-level policy with state compression: learns compressed robot state representation
class StochasticLowLevelPolicyNetworkWithCompression(torch.nn.Module):
    def __init__(self, args):
        super(StochasticLowLevelPolicyNetworkWithCompression, self).__init__()
        self.args = args

        self.embedding_layer = torch.nn.Embedding(self.args.number_policies, self.args.robot_state_dim + 1)

        # Project skill parameters to compression vector
        self.projection_layer1 = torch.nn.Linear(self.args.z_dimensions, 128)
        self.projection_layer2 = torch.nn.Linear(128, self.args.robot_state_dim + 1)
        # Layers
        self.state_compression_layer1 = torch.nn.Linear(1 + self.args.z_dimensions, self.args.hidden_size_compression)
        self.state_compression_layer2 = torch.nn.Linear(args.hidden_size_compression, args.hidden_size_compression)

        self.correction_layer1 = torch.nn.Linear(args.state_dim + self.args.hidden_size_compression \
                                                 + self.args.z_dimensions, self.args.hidden_size_compression)
        self.correction_layer2 = torch.nn.Linear(args.hidden_size_compression, args.hidden_size_compression)
        # Initialize action decoder
        self.pre_GMM_layer = torch.nn.Linear(1 + args.z_dimensions + args.hidden_size_compression, self.args.gmm_hidden_size)
        # network_kwargs = {'hidden_size': args.gmm_hidden_size, 'num_layers': args.gmm_num_layers, 'min_std': 0.0001,
        #                   'num_modes': args.num_gaussians, 'low_eval_noise': False, 'activation': 'softplus',
        #                   'input_size': args.gmm_hidden_size, 'output_size': args.action_dim}
        # loss_kwargs = {'loss_coef': 1.0}

        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size_primitive, args.hidden_size_primitive),
            nn.ReLU(),
            nn.Linear(args.hidden_size_primitive, args.hidden_size_primitive),
            nn.ReLU(),
            nn.Linear(args.hidden_size_primitive, args.action_dim)
        )
        # self.GMM_head = GMMHead(**network_kwargs, **loss_kwargs)
        # Activation funcs
        self.hidden_activation = torch.nn.ReLU()
        self.activation_layer = torch.nn.Tanh()

    def forward(self, input, robot_state, z_vals, skill_idx):
        assert self.args.markov_subpolicy  # Currently only supports Markov policies
        if self.args.mask_robot_state:
            robot_state = robot_state.clone()
            robot_state[:, :, 4:] = 0
        # Compress robot state using learned projection based on skill parameters
        # discrete_embeddings = self.embedding_layer(skill_idx)
        # discrete_embeddings = torch.zeros(discrete_embeddings.shape).to(device)
        # projection_vectors = torch.cat((discrete_embeddings, z_vals), dim = -1)

        # Calculate the 1D compressed state. See Appendix A for a note on why the discrete skill index is not used here.
        projection_vectors = self.hidden_activation(self.projection_layer1(z_vals))
        projection_vectors = self.projection_layer2(projection_vectors)
        constants = torch.ones((robot_state.shape[0], robot_state.shape[1], 1)).to(device)
        compressed_robot_state = torch.sum((projection_vectors * torch.cat((robot_state, constants), dim = -1)), dim = -1).reshape(input.shape[0], input.shape[1], 1)
        compressed_robot_state = self.activation_layer(compressed_robot_state)

        # Predict action as a function of the compressed state and the continuous parameter
        cae_inputs = torch.cat((compressed_robot_state, z_vals), dim = -1)
        compressed_action_embedding = self.activation_layer(self.state_compression_layer1(cae_inputs))
        compressed_action_embedding = self.state_compression_layer2(compressed_action_embedding)

        # Train a residual network to "correct" the predicted action while having access to the full input image. THIS IS NOT USED.
        correction_layer_inputs = torch.cat((input, z_vals.detach(), compressed_action_embedding.detach()), dim = -1)
        corrected_action_embedding = self.activation_layer(self.correction_layer1(correction_layer_inputs))
        corrected_action_embedding = self.correction_layer2(corrected_action_embedding)

        dist = self.decoder(compressed_action_embedding) #self.GMM_head(compressed_action_embedding)
        corrected_dist = self.decoder(compressed_action_embedding.detach() + corrected_action_embedding) #self.GMM_head(compressed_action_embedding.detach() + corrected_action_embedding)

        # if you remove residual network, make sure resner gets gradients from elsewhere
        return corrected_dist, dist

    def get_actions(self, input, robot_state, z_vals, skill_idx):
        with torch.no_grad():
            corrected_dist, dist = self.forward(input, robot_state, z_vals, skill_idx)
            corrected_action, action = corrected_dist.detach(), dist.detach()
        return corrected_action, action

# Discriminator network: predicts task ID from skill embeddings (for adversarial regularization)
class TaskPredictionMLP(torch.nn.Module):
    def __init__(self, args):

        super(TaskPredictionMLP, self).__init__()
        self.layer1 = torch.nn.Linear(args.z_dimensions, 520)
        self.layer2 = torch.nn.Linear(520, 520)
        self.layer3 = torch.nn.Linear(520, 90)
        self.hidden_activation = torch.nn.ReLU()

    def forward(self, input):
        output = self.hidden_activation(self.layer1(input))
        output = self.hidden_activation(self.layer2(output))
        output = self.layer3(output)
        return output

# Data augmentation and image encoder
translation_aug = TranslationAug(input_shape = [3, 84, 84], translation = 4)
IMG_AUG = DataAugGroup((translation_aug,))
IMAGE_ENCODER = Encoder(obs_shape=(3, 84, 84), feature_dim = 64).to(device)


