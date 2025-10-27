"""
This file contains all neural modules related to encoding the spatial
information of obs_t, i.e., the abstracted knowledge of the current visual
input conditioned on the language.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from robomimic.models.base_nets import CropRandomizer


###############################################################################
#
# Modules related to encoding visual information (can conditioned on language)
#
###############################################################################
class DataAugGroup(nn.Module):
    """
    Add augmentation to multiple inputs
    """
    def __init__(self, aug_list):
        super().__init__()
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, x_groups):
        split_channels = []
        for i in range(len(x_groups)):
            split_channels.append(x_groups[i].shape[1])
        x = torch.cat(x_groups, dim=1)
        out = self.aug_layer(x)
        out = torch.split(out, split_channels, dim=1)
        return out

class BatchWiseImgColorJitterAug(torch.nn.Module):
    """
    Color jittering augmentation to individual batch.
    This is to create variation in training data to combat
    BatchNorm in convolution network.
    """
    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        out = []
        for x_i in torch.split(x, 1):
            if np.random.rand() > self.epsilon:
                out.append(self.color_jitter(x_i))
            else:
                out.append(x_i)
        return torch.cat(out, dim=0)

    def output_shape(self, input_shape):
        return input_shape
class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """

    def __init__(
        self,
        input_shape,
        translation,
    ):
        super().__init__()

        self.pad_translation = translation // 2
        pad_output_shape = (
            input_shape[0],
            input_shape[1] + translation,
            input_shape[2] + translation,
        )

        self.crop_randomizer = CropRandomizer(
            input_shape=pad_output_shape,
            crop_height=input_shape[1],
            crop_width=input_shape[2],
        )

    def forward(self, x):
        batch_size, temporal_len, img_c, img_h, img_w = x.shape
        x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
        out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
        out = self.crop_randomizer.forward_in(out)
        out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)
        return out

    def output_shape(self, input_shape):
        return input_shape

class PatchEncoder(nn.Module):
    """
    A patch encoder that does a linear projection of patches in a RGB image.
    """

    def __init__(
        self, input_shape, patch_size=[16, 16], embed_size=64, no_patch_embed_bias=False
    ):
        super().__init__()
        C, H, W = input_shape
        num_patches = (H // patch_size[0] // 2) * (W // patch_size[1] // 2)
        self.img_size = (H, W)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.h, self.w = H // patch_size[0] // 2, W // patch_size[1] // 2

        self.conv = nn.Sequential(
            nn.Conv2d(
                C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            64,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )
        self.bn = nn.BatchNorm2d(embed_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.proj(x)
        x = self.bn(x)
        return x


# class ResnetEncoder(nn.Module):
#     """
#     A Resnet-18-based encoder for mapping an image to a latent vector
#
#     Encode (f) an image into a latent vector.
#
#     y = f(x), where
#         x: (B, C, H, W)
#         y: (B, H_out)
#
#     Args:
#         input_shape:      (C, H, W), the shape of the image
#         output_size:      H_out, the latent vector size
#         pretrained:       whether use pretrained resnet
#         freeze: whether   freeze the pretrained resnet
#         remove_layer_num: remove the top # layers
#         no_stride:        do not use striding
#     """
#
#     def __init__(
#         self,
#         input_shape,
#         output_size,
#         pretrained=False,
#         freeze=False,
#         remove_layer_num=2,
#         no_stride=False,
#         language_dim=768,
#         language_fusion="film",

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.repr_dim = feature_dim

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return self.trunk(h)

# try:
#      - less diverse demonstrations
#      - traiing only on robot state - this will tll you whether the issue is in the image encoder or the demonstrations
#       - try  multiple gaussians in the gmm