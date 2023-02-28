import numpy as np
import cv2
import glob
from tqdm import tqdm
import wandb
from einops import rearrange
from unet import Unet
import torch
import os
from sklearn.model_selection import train_test_split
from copy import deepcopy


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


    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image


w = 512
h = 512
max_time = 5

image_paths = glob.glob('[path to rbg images]/*.[file type]')
depth_map = glob.glob('[path to depth images]/*.[file type]')

data = []
sample_image = None

for i in range(5):# range(len(image_paths)):
    img = cv2.imread(image_paths[i])
    depthmap = cv2.imread(depth_map[i], cv2.IMREAD_GRAYSCALE)

    # resize image and depthmap to desired dimensions
    img_resized = cv2.resize(img, (w, h))
    img_resized = np.divide(img_resized, 255.)
    img_resized = rearrange(img_resized, 'h w c -> c h w')
    if sample_image is None:
        sample_image = img_resized
    depthmap_resized = cv2.resize(depthmap, (w, h))
    depthmap_resized = np.divide(depthmap_resized, 255.)

    t = max_time

    for i in range(t):
        if t-i == 1:
            prev_depthmap = np.ones_like(depthmap_resized)
        else:
            prev_depthmap = apply_gaussian_noise(depthmap_resized, 1)
        stacked = np.concatenate((img_resized, rearrange(np.expand_dims(prev_depthmap, -1), 'h w c -> c h w')), axis=0)
        data.append((stacked, rearrange(np.expand_dims(depthmap_resized, -1), 'h w c -> c h w'), t-1))
        depthmap_resized = prev_depthmap

np.random.shuffle(data)
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
img, depth, times = zip(*train_data)
img_val, depth_val, times_val = zip(*val_data)


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


name = "depth_diffusion"
path = "diffusion/"

if not os.path.exists(path + name + "/"):
    os.mkdir(path+name)
    os.mkdir(path+name+"/"+"checkpoints")

model_save_path = path + name + "/" + f"{name}.pth"
model_state_save_path = path + name + "/" + f"{name}_state.pth"
checkpts_save_path = path + name + "/" + "checkpoints"

wandb.init(project="depth_diffusion", name="diffusion_v1")

batch_size = 64
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(device).to(device)
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

train_data, train_batches = batchify(img, depth, times, batch_size)
val_data, val_batches = batchify(img_val, depth_val, times_val, batch_size)

test_img = rearrange(sample_image * 255.0, 'c h w -> h w c')
wandb.log({"examples": [wandb.Image(test_img, caption="example image original")]})
test_img_input = sample_image

wandb.watch(model, mse_loss, log="all", log_graph=True)

for epoch in range(num_epochs):

    model.train()

    pbar = tqdm(range(train_batches), desc=f"Epoch {epoch}")

    train_loss = 0

    for i in pbar:
        x = torch.from_numpy(train_data[0][i])
        y = torch.from_numpy(train_data[1][i])
        t = torch.from_numpy(train_data[2][i])
        x = x.to(device=device, dtype=torch.float)
        y = y.to(device=device, dtype=torch.float)
        t = t.to(device=device, dtype=torch.float)

        optimizer.zero_grad()

        y_pred = model(x, t)

        loss = mse_loss(y_pred, y)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        train_loss = loss.item()

        loss.backward()
        optimizer.step()
    
    model.eval()

    with torch.no_grad():
        x = np.concatenate((test_img_input, np.ones_like(depth[0])), axis=0)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).to(device=device, dtype=torch.float)

        for i in range(max_time):
            time_input = np.array([i])
            time_input = torch.from_numpy(time_input).to(device=device, dtype=torch.float)
            test_img_pred = model(x, time_input)
            x = np.concatenate((test_img_input, test_img_pred.detach().cpu().numpy()[0]), axis=0)
            x = np.expand_dims(x, axis=0)
            x = torch.from_numpy(x).to(device=device, dtype=torch.float)
        
        test_img_rearranged = test_img_pred.detach().cpu().numpy()[0] * 255.0
        test_img_rearranged = rearrange(test_img_rearranged, 'c h w -> h w c')
        wandb.log({"outputs": [wandb.Image(test_img_rearranged, caption=f"depth at epoch {epoch}")]})

        val_loss_num = 0

        for i in range(val_batches):
            x = torch.from_numpy(val_data[0][i])
            y = torch.from_numpy(val_data[1][i])
            t = torch.from_numpy(val_data[2][i])
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            t = t.to(device=device, dtype=torch.float)

            y_pred = model(x, t)

            val_loss = mse_loss(y_pred, y)

            val_loss_num = val_loss.item()
        
        print(f"Eval: val_loss:{val_loss_num}")

        wandb.log({"train":{"loss":train_loss}, "val":{"loss":val_loss_num}})
    
    if epoch % 10 == 9:
        print(f"Saving checkpoint at epoch {epoch}")
        torch.save(deepcopy(model.state_dict()), checkpts_save_path + f"/epoch_{epoch}_chkpt.pth")
        torch.save(model.state_dict(), model_state_save_path)
        torch.save(model, model_save_path)


torch.save(model.state_dict(), model_state_save_path)
torch.save(model, model_save_path)
