import argparse

# Parse command-line arguments for training configuration
def parse_arguments():
    parser = argparse.ArgumentParser(description='Learning Skills from Demonstrations')

    # Basic args
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--finetune', dest='finetune', type=int, default=0)
    parser.add_argument('--algo', dest='algo', type=str, default=None)
    parser.add_argument('--debug', dest='debug', type=int, default=0)
    parser.add_argument('--name', dest='name', type=str, default="test")
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=3)
    parser.add_argument('--model', dest='model', type=str)
    parser.add_argument('--logdir', dest='logdir', type=str, default='/oscar/data/gdk/paraskill')
    parser.add_argument('--epochs', dest='epochs', type=int, default=400)
    parser.add_argument('--seed', dest='seed', type=int, default=95)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=3e-4)

    # Network size args
    parser.add_argument('--number_layers', dest='number_layers', type=int, default=2)
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256)
    parser.add_argument('--hidden_size_primitive', dest='hidden_size_primitive', type=int, default=64)
    parser.add_argument('--hidden_size_compression', dest='hidden_size_compression', type=int, default=1024)
    parser.add_argument('--gmm_num_layers', dest='gmm_num_layers', type=int, default=2)
    parser.add_argument('--gmm_hidden_size', dest='gmm_hidden_size', type=int, default=1024)
    parser.add_argument('--num_gaussians', dest='num_gaussians', type=int, default=1)
    parser.add_argument('--subpolicy_num_layers', dest='subpolicy_num_layers', type=int, default=1)
    parser.add_argument('--variational_latent_num_layers', dest='variational_latent_num_layers', type=int, default=2)
    parser.add_argument('--test_dataset', dest='test_dataset', type=str, default="libero_10")
    parser.add_argument('--train_dataset', dest='train_dataset', type=str, default="libero_90")
    parser.add_argument('--lib90_split_idx', dest='lib90_split_idx', type=int, default=80)

    # Param Skill args
    parser.add_argument('--z_dimensions', dest='z_dimensions', type=int, default=2)
    parser.add_argument('--number_policies', dest='number_policies', type=int, default=10)
    parser.add_argument('--add_noise_to_z', dest='add_noise_to_z', type=int, default=1)
    parser.add_argument('--variance_factor', dest='variance_factor', type=int, default=100)
    parser.add_argument('--center_loss_weight', dest='center_loss_weight', type=float, default=0.1)
    parser.add_argument('--kl_loss_weight', dest='kl_loss_weight', type=float, default=0.1)
    parser.add_argument('--z_loss_weight', dest='z_loss_weight', type=float, default=0.01)
    parser.add_argument('--variance_reg_weight', dest='variance_reg_weight', type=float, default=0.)
    parser.add_argument('--discriminator_loss_weight', dest='discriminator_loss_weight', type=float, default=0.)
    parser.add_argument('--mask_latent_to_var', dest='mask_latent_to_var', type=int, default=0)
    parser.add_argument('--variational_z_mean_nonlinearity', dest='variational_z_mean_nonlinearity', type=int, default=0)
    parser.add_argument('--restrict_state_range', dest='restrict_state_range', type=int, default=1)


    # Env args
    parser.add_argument('--state_dim', dest='state_dim', type=int, default=169)
    parser.add_argument('--robot_state_dim', dest='robot_state_dim', type=int, default=9)
    parser.add_argument('--action_dim', dest='action_dim', type=int, default=7)
    parser.add_argument('--traj_length', dest='traj_length', type=int, default=517)#388)#373)
    parser.add_argument('--input_size', dest='input_size', type=int, default=176)
    parser.add_argument('--language_dim', dest='language_dim', type=int, default=768)
    parser.add_argument('--libero_root_path', dest='libero_root_path', type=str, default='/oscar/data/gdk/vgupta17/LIBERO/')

    # Finetune args
    parser.add_argument('--finetune_subpolicy', dest='finetune_subpolicy', type=int, default=1)
    parser.add_argument('--finetune_resnet', dest='finetune_resnet', type=int, default=0)
    parser.add_argument('--three_shot_finetune', dest='three_shot_finetune', type=int, default=0)
    parser.add_argument('--use_corrected_action', dest='use_corrected_action', type=int, default=1)
    parser.add_argument('--downstream_rollout_frequ', dest='downstream_rollout_frequ', type=int, default=1)

    # Subpolicy args
    parser.add_argument('--markov_subpolicy', dest='markov_subpolicy', type=int, default=0)
    parser.add_argument('--image_to_subpolicy', dest='image_to_subpolicy', type=int, default=0)
    parser.add_argument('--use_language', dest='use_language', type=int, default=0)
    parser.add_argument('--apply_aug', dest='apply_aug', type=int, default=0)

    return parser.parse_args()

