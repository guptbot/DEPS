import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

# Combined distribution: GMM for continuous actions + categorical for gripper
class GMMWithBinary(D.Distribution):
    def __init__(self, gmm, binary_dist):
        super().__init__()
        self.gmm = gmm
        self.binary_dist = binary_dist

    def sample(self):
        gmm_sample = self.gmm.sample()
        binary_sample = self.binary_dist.sample()
        # Convert 0/1 to -1/1 for Libero gripper action format
        binary_sample = binary_sample * 2 - 1
        return torch.cat([gmm_sample, binary_sample.unsqueeze(-1)], dim=-1)

    def log_prob(self, value):
        gmm_log_prob = self.gmm.log_prob(value[..., :-1])
        # Convert -1/1 back to 0/1 for probability calculation
        binary_vars = (value[..., -1] + 1) / 2
        # Handle zero padding (converts 0 back to 1)
        binary_vars[binary_vars == 0.5] = 1
        binary_log_prob = self.binary_dist.log_prob(binary_vars)
        return gmm_log_prob + binary_log_prob

# Action head: outputs GMM for continuous actions and categorical for gripper
class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()
        self.mean_layer = nn.Linear(hidden_size, (output_size - 1) * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, (output_size - 1) * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)
        self.binary_logits_layer = nn.Linear(hidden_size, 2)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size - 1)
        means = torch.tanh(means)
        logits = self.logits_layer(share)
        binary_logits = self.binary_logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size - 1
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits, binary_logits

    def forward(self, x):
        # Apply network to each timestep if input is 3D
        if x.ndim == 3:
            means, scales, logits, binary_logits = TensorUtils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits, binary_logits = self.forward_fn(x)

        # Create GMM distribution for continuous actions
        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        # Create categorical distribution for gripper open/close
        binary_dist = D.Categorical(logits=binary_logits)
        combined_dist = GMMWithBinary(gmm, binary_dist)
        return combined_dist