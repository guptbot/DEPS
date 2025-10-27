
import torch
from PIL import Image, ImageDraw, ImageFont
from utils.logger import Logger
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams['animation.bitrate'] = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Overlay skill indices and parameters on trajectory images for visualization
def visualize_traj_partitioning(batch, k, z, train, step):
    images = batch['obs']['agentview_rgb']
    font = ImageFont.load_default()
    traj_img = images[0]
    traj_k = k[0]
    traj_z = z[0]
    video_images = []
    for image, k_val, z_val in zip(traj_img, traj_k, traj_z): # iterate over images
        if isinstance(image, torch.Tensor):
            image = torch.flip(image, [1]).permute(1, 2, 0).cpu().numpy()  # Change from CxHxW to HxWxC
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        text = str(k_val.item()) + "\n" + str("\n".join(map(str, z_val.flatten().tolist())))
        position = (10, 10)
        draw.text(position, text, font=font, fill=(255, 255, 255))
        video_image = np.array(image)
        video_image = np.transpose(video_image, (2, 0, 1))  # Convert to CxHxW
        video_images.append(video_image)
    video_images = np.stack(video_images)
    key = "traj partitioning train" if train else "traj partitioning test"
    Logger.log({key: Logger.create_video(video_images, fps=40, format="mp4")}, step=step)

# Create scatter plot of 2D skill embeddings colored by task
def store_scatter(data, color_map, marker_map, step):
    data = data.detach().cpu().numpy()
    x = data[..., 0]
    y = data[..., 1]
    plt.figure(figsize=(10, 6))
    markers = list(matplotlib.markers.MarkerStyle.markers.keys())[:50]
    colors = ['blue', 'red', 'green']
    assert len(set(color_map)) == len(set([i % 3 for i in color_map]))  # make sure different tasks are not mapped to the same color
    for i in range(len(x)):
        color = colors[color_map[i] % 3]
        marker = markers[marker_map[i]]
        plt.scatter(x[i], y[i], color=color, marker=marker, alpha=0.8)
    plt.title('Scatter Plot of first two components of continuous parameters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    fig = plt.gcf()
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)
    image = Logger.create_image(image, caption="test")
    Logger.log({"Scatter plt": image}, step=step)


# Save rollout trajectory as video for WandB logging
def save_video(tensor_list, task_idx, step, fps=40):
    np_imgs = []
    for tensor in tensor_list:
        # Convert the tensor to a NumPy array
        frame = torch.flip((tensor * 255), [1]).cpu().numpy().astype('uint8')
        np_imgs.append(frame)
    np_imgs = np.stack(np_imgs, axis = 0)
    Logger.log({f"rollout_task_{task_idx}": Logger.create_video(np_imgs, fps=fps, format="mp4"), f"Task {task_idx} epoch": step})



