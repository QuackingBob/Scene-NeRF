import random
import numpy as np
import torch
from tqdm import tqdm

def sample_gaussian(mean, stddev):
    return random.gauss(mean, stddev)

def sample_uniform(low, high):
    return random.uniform(low, high)

def stratified_quadrature_sampling(t_n, t_f, N, density_func, color_func, ray_func, d):
    T_i_sum = 0
    C_r = np.zeros((3))
    t_i = 0
    for i in range(N):
        t_i_next = sample_uniform(t_n + (i - 1) / N * (t_f - t_n), t_n + i / N * (t_f - t_n))
        delta_i = t_i_next - t_i
        t_i = t_i_next 
        r_i = ray_func(t_i)
        density_sample = density_func(r_i)
        color_sample = color_func(r_i, d)
        T_i_sum += density_sample * delta_i
        T_i = np.exp(-1 * T_i_sum)
        C_r += T_i * (1 - np.exp(-1 * density_sample * delta_i)) * color_sample
    return C_r


def generate_image(model, pose, resolution=512, n_samples_coarse=64, n_samples_fine=128, device='cpu', cone_aperture=20):
    """
    Generates an image by sampling a neural radiance field (NeRF) model with a random pose using hierarchical sampling.

    Args:
        model (torch.nn.Module): The NeRF model to sample.
        pose (np.ndarray): A 4x4 homogeneous transformation matrix representing the pose of the camera.
        resolution (int): The output image resolution (default=512).
        n_samples_coarse (int): The number of coarse samples to take per pixel (default=64).
        n_samples_fine (int): The number of fine samples to take per pixel (default=128).
        device (str): The device to use for the model (default='cpu').
        cone_aperture (float): The aperture angle (in degrees) of the cone projection (default=20).

    Returns:
        The rendered image as a numpy array with shape (resolution, resolution, 3).
    """
    # Generate the rays
    rays = generate_rays(pose, resolution, cone_aperture)

    # Sample the rays hierarchically
    samples = sample_rays_hierarchical(model, rays, n_samples_coarse, n_samples_fine, device)

    # Render the final image
    image = render_image(samples)

    return image


def generate_rays(pose, resolution, cone_aperture):
    """
    Generates a cone projection of rays.

    Args:
        pose (np.ndarray): A 4x4 homogeneous transformation matrix representing the pose of the camera.
        resolution (int): The output image resolution.
        cone_aperture (float): The aperture angle (in degrees) of the cone projection.

    Returns:
        A numpy array with shape (resolution, resolution, 6) containing the ray origins and directions for each pixel.
    """
    # Compute the pixel coordinates in the image plane
    x, y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    pixel_coords = np.stack([x, y, np.ones_like(x)], axis=-1)

    # Compute the ray directions and origins
    ray_directions = np.matmul(pixel_coords, np.linalg.inv(pose[:3, :3]).T)
    ray_origins = np.tile(pose[:3, 3], (resolution, resolution, 1))
    ray_origins += ray_directions * 0.1

    # Generate a cone projection of rays
    ray_directions /= np.linalg.norm(ray_directions, axis=-1, keepdims=True)
    aperture_radians = np.deg2rad(cone_aperture)
    cone_direction = np.array([0, 0, -1])
    cone_angle = np.arccos(np.dot(cone_direction, ray_directions))
    ray_mask = cone_angle < aperture_radians
    rays = np.concatenate([ray_origins, ray_directions], axis=-1)
    rays = rays[ray_mask]

    return rays


def sample_rays_hierarchical(model, rays, n_samples_coarse, n_samples_fine, device):
    """
    Samples the given rays using hierarchical sampling.

    Args:
        model (torch.nn.Module): The NeRF model to sample.
        rays (np.ndarray): A numpy array with shape (num_rays, 6) containing the ray origins and directions.
        n_samples_course: An integer, the number of course samples (course model) for hierarchical sampling
        n_samples_fine":  An integer, the number of fine samples (fine model) for hierarchical sampling
        device: str (cpu or gpu)
    """
    n_rays = rays.shape[0]
    samples = np.zeros((n_rays, n_samples_coarse + n_samples_fine, 4), dtype=np.float32)

    # Course sampling
    with torch.no_grad():
        rays_t = torch.tensor(rays, dtype=torch.float32, device=device)
        coarse_z = model(rays_t[:, :3])
        coarse_z = coarse_z.cpu().numpy()
        coarse_z = np.clip(coarse_z, 1e-5, 1 - 1e-5)

    coarse_samples = np.random.uniform(size=(n_rays, n_samples_coarse))
    coarse_samples = np.sort(coarse_samples, axis=-1)
    coarse_samples = np.concatenate([coarse_samples, np.ones((n_rays, 1))], axis=-1)
    coarse_samples = -np.log(1 - coarse_samples) / coarse_z[:, None]
    coarse_samples = np.cumsum(coarse_samples, axis=1)
    coarse_samples = np.concatenate([np.zeros((n_rays, 1)), coarse_samples], axis=-1)
    coarse_samples = coarse_samples[:, :-1]
    coarse_samples = np.stack([coarse_samples, np.zeros_like(coarse_samples)], axis=-1)

    samples[:, :n_samples_coarse] = coarse_samples

    # Fine sampling
    with torch.no_grad():
        fine_samples = []
        for i in tqdm(range(0, n_samples_fine, 4096), desc='Fine sampling', leave=False):
            j = min(i + 4096, n_samples_fine)
            rays_t = torch.tensor(np.concatenate([rays, samples[:, :n_samples_coarse]], axis=-1), dtype=torch.float32, device=device)
            fine_z = model(rays_t[:, :3])
            fine_z = fine_z.cpu().numpy()
            fine_z = np.clip(fine_z, 1e-5, 1 - 1e-5)
            fine_samples_i = np.random.uniform(size=(n_rays, j))
            fine_samples_i = np.sort(fine_samples_i, axis=-1)
            fine_samples_i = np.concatenate([fine_samples_i, np.ones((n_rays, 1))], axis=-1)
            fine_samples_i = -np.log(1 - fine_samples_i) / fine_z[:, None]
            fine_samples_i = np.cumsum(fine_samples_i, axis=1)
            fine_samples_i = np.concatenate([samples[:, :n_samples_coarse], fine_samples_i], axis=-1)
            fine_samples_i = np.sort(fine_samples_i, axis=-1)
            fine_samples_i = fine_samples_i[:, n_samples_coarse:]
            fine_samples.append(fine_samples_i)

        fine_samples = np.concatenate(fine_samples, axis=-1)
        samples[:, n_samples_coarse:] = fine_samples

    return samples
