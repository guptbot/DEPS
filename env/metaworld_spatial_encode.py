import torch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def spatial_encode(batch, use_language, image_encoder):
    assert (not use_language)

    imgs = batch['imgs'].float()
    robot_states = batch['robot_states'].float()

    encoded = []

    # 1: Encode Image

    B, T, C, H, W = imgs.shape
    imgs = imgs.reshape(-1, C, H, W)
    encoded_imgs = image_encoder(imgs)
    encoded_imgs = encoded_imgs.reshape(B, T, -1)
    #encoded_imgs = torch.zeros(encoded_imgs.shape).to(device)

    # 2: Get robot state
    robot_state = batch['robot_states'].float()


    encoded = torch.cat([encoded_imgs, robot_state], dim=-1)

    return encoded, robot_state


if __name__ == "__main__":
    from env.metaworld_dataloader import fetch_dataloader
    for data in fetch_dataloader():
        spatial_encode(data)
        break


