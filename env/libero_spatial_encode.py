import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from einops import rearrange, repeat
#from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *
from networks.resnet import ResnetEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract low-dimensional proprioceptive features (joint states, gripper)
class ExtraModalities:
    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):

        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3

        self.extra_low_level_feature_dim = (
            int(use_joint) * joint_states_dim
            + int(use_gripper) * gripper_states_dim
            + int(use_ee) * ee_dim
        )
        assert self.extra_low_level_feature_dim > 0, "[error] no extra information"

    def __call__(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []
        if self.use_joint:
            tensor_list.append(obs_dict["joint_states"])
        if self.use_gripper:
            tensor_list.append(obs_dict["gripper_states"])
        if self.use_ee:
            tensor_list.append(obs_dict["ee_states"])
        x = torch.cat(tensor_list, dim=-1)
        return x

    def output_shape(self, input_shape, shape_meta):
        return (self.extra_low_level_feature_dim,)
EXTRA_ENCODER = ExtraModalities(use_joint = True, use_gripper = True, use_ee = False)

# Encode observations into feature vectors: images + robot state + language
def spatial_encode(data, use_language, image_encoders, language_encoder):
    # 1. Encode images from multiple camera views
    encoded = []
    img_angles = [key for key in data['obs'].keys() if 'rgb' in key]
    langs = data["task_emb"]
    if not use_language:
        langs = torch.zeros(langs.shape).to(device)
    for img_name in img_angles:
        x = data["obs"][img_name]
        B, T, C, H, W = x.shape
        e = image_encoders[img_name](
            x.reshape(B * T, C, H, W),
            langs=langs
            .reshape(B, 1, -1)
            .repeat(1, T, 1)
            .reshape(B * T, -1),
        ).view(B, T, -1)
        encoded.append(e)
    # 2. Add proprioceptive state (joint angles, gripper position)
    encoded.append(EXTRA_ENCODER(data["obs"]))  # add (B, T, H_extra)
    encoded = torch.cat(encoded, -1)  # (B, T, H_all)

    robot_state = EXTRA_ENCODER(data["obs"])  # Separate robot state for compression


    # 3. Encode and append language instruction
    lang_h = language_encoder(data)  # (B, H)
    if not use_language:
        lang_h = torch.zeros(lang_h.shape).to(device)
    encoded = torch.cat(
        [encoded, lang_h.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1
    )

    return encoded, robot_state

if __name__ == "__main__":
    from env.libero_dataloader import fetch_dataloader
    for data in fetch_dataloader():
        spatial_encode(data)
        break


