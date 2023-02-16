import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from model import Generator


def interpolate(z_1, z_2, n_steps=10):
    """Performs Linear Interpolation between Two Latent Vectors

    Args:
        z_1 (Tensor): Latent vector 1
        z_2 (Tensor): Latent vector 2
        n_steps (int, optional): Number of steps. Defaults to 10.

    Returns:
        Tensor: Latent Points between z_1 and z_2
    """
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n_steps)])
    return z


parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='saved_models/model_final.pth',
                    help='Checkpoint to load path from')
parser.add_argument('-num_output', default=32,
                    help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path, map_location=torch.device('cpu'))

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(
    latent_size=params['nz'], n_channels=params['nc'], features_g=params['ngf']).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])
print(netG)

print(args.num_output)
# Get latent vector Z from unit normal distribution.
noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)
# Interpolating between two latent vectors
interpolate_pts = interpolate(noise[0], noise[1])

with torch.no_grad():
    generated_img = netG(noise).detach().cpu()
    interpolate_img = netG(interpolate_pts).detach().cpu()

vutils.save_image(generated_img.data,
                  'results_local/generated_images.png', normalize=True)
grid_img = vutils.make_grid(interpolate_img, nrow=10)
vutils.save_image(
    grid_img.data, 'results_local/interpolated_images.png', normalize=True)
