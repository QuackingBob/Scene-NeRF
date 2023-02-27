import numpy as np
import cv2
import glob
import tqdm
import wandb
from einops import rearrange

def apply_gaussian_noise(image, num_times, mean=0, std=1):
    """
    Applies Gaussian noise to a grayscale image a specified number of times.
    """
    noisy_image = image.copy()
    height, width = image.shape[:2]

    for i in range(num_times):
        # Generate Gaussian noise with the specified mean and standard deviation
        noise = np.random.normal(mean, std, size=(height, width))

        # Add the noise to the image
        noisy_image = cv2.addWeighted(noisy_image, 0.5, noise, 0.5, 0)


    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


w = 512
h = 512
max_time = 5

image_paths = []
depth_map = []

data = []

for i in range(len(image_paths)):
    img = cv2.imread(image_paths[i])
    depthmap = cv2.imread(y[i])

    # resize image and depthmap to desired dimensions
    img_resized = cv2.resize(img, (w, h))
    img_resized = rearrange(img_resized, 'h w c -> c h w')
    depthmap_resized = cv2.resize(depthmap, (w, h))

    t = max_time

    for i in range(t):
        if t-i == 1:
            prev_depthmap = np.zeros_like(depthmap_resized)
        else:
            prev_depthmap = apply_gaussian_noise(depthmap_resized, 1)   
        stacked = np.concatenate((img_resized, prev_depthmap), axis=2)
        data.append((img, rearrange(depthmap_resized, 'h w c -> c h w'), t))
        depthmap_resized = prev_depthmap

np.random.shuffle(data)
img, depth, times = zip(*data)


def batchify(x, y, t, batch_size):
    num_batches = len(x) // batch_size
    x_batch = []
    y_batch = []
    t_batch = []
    for i in tqdm(range(num_batches), desc="Creating Batches"):
        x_batch.append(np.stack(x[i*batch_size:(i+1)*batch_size], axis=0))
        y_batch.append(np.stack(y[i*batch_size:(i+1)*batch_size], axis=0))
        t_batch.append(np.stack(t[i*batch_size:(i+1)*batch_size], axis=0))
    return (x_batch, y_batch, t_batch), num_batches

