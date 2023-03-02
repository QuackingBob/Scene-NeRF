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
import h5py


w = 512
h = 512
max_time = 5

data_dir = "[specify path to dataset]"

image_depth_fname_dict = {}
sample_image = None

for root, dirs, files in os.walk(data_dir):
    for file_name in files:
        if file_name.endswith(".jpg"):
            img_path = os.path.join(root, file_name)
            depth_path = os.path.join(root, "depthmap", file_name[:-3] + ".h5")
            image_depth_fname_dict[img_path] = depth_path
            if sample_image is None:
                img_temp = cv2.imread(img_path)
                img_temp = cv2.resize(img_temp, (w, h))
                img_temp = np.divide(img_temp, 255.)
                img_temp = rearrange(img_temp, 'h w c -> c h w')
                sample_image = img_temp


batch_size = 64
num_epochs = 100

def batchify(data_dict, data_keys_shuffled, batch_size):
    x = data_keys_shuffled
    num_batches = len(x) // batch_size
    batches = []
    for i in tqdm(range(num_batches), desc="Creating Batches"):
        batches.append(([x[j] for j in range(i*batch_size,(i+1)*batch_size)], [data_dict[x[j]] for j in range(i*batch_size,(i+1)*batch_size)]))
    return batches, num_batches

image_keys = list(image_depth_fname_dict.keys())

train_batches, val_batches = train_test_split(image_keys, test_size=0.1, random_state=42)
train_data, num_train_batches = batchify(image_depth_fname_dict, train_batches, batch_size)
val_data, num_val_batches = batchify(image_depth_fname_dict, val_batches, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="depth_diffusion", name="diffusion_v1")

def openh5file(path):
    file_h5 = h5py.File(path, 'r')
    depth_map = file_h5["depth"]
    file_h5.close()
    return depth_map


def apply_gaussian_noise(image, num_times, mean=0, std=1):
    """
    Applies Gaussian noise to a grayscale image a specified number of times.
    """
    noisy_image = image.copy()
    height, width = image.shape[:2]

    for i in range(num_times):
        noise = np.random.normal(mean, std, size=(height, width))

        noisy_image = cv2.addWeighted(noisy_image, 0.5, noise, 0.5, 0)


    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image


def load_batch(batch):
    img_pth, depth_path = batch
    num = batch_size
    t = np.random.randint(0, max_time, size=num)

    images = []
    depths = []

    for i in range(batch_size):
        img = cv2.imread(img_pth[i])
        depth_map = openh5file(depth_path[i])

        img_resized = cv2.resize(img, (w, h))
        img_resized = np.divide(img_resized, 255.)
        img_resized = rearrange(img_resized, 'h w c -> c h w')
        depthmap_resized = cv2.resize(depth_map, (w, h))

        noisy_depthmap = apply_gaussian_noise(depthmap_resized, int(max_time - t[i]))
        noisy_depthmap_next = apply_gaussian_noise(depthmap_resized, int(max_time - t[i]) - 1)
        
        stacked = np.concatenate((img_resized, rearrange(np.expand_dims(noisy_depthmap, -1), 'h w c -> c h w')), axis=0)
        
        images.append(stacked)
        depths.append(rearrange(np.expand_dims(noisy_depthmap_next, -1), 'h w c -> c h w'))
    
    images = np.stack(images)
    depths = np.stack(depths)
    t = np.divide(t, max_time)

    return (images, depths, t)

name = "depth_diffusion"
path = "diffusion/"

if not os.path.exists(path + name + "/"):
    os.mkdir(path+name)
    os.mkdir(path+name+"/"+"checkpoints")

model_save_path = path + name + "/" + f"{name}.pth"
model_state_save_path = path + name + "/" + f"{name}_state.pth"
checkpts_save_path = path + name + "/" + "checkpoints"

model = Unet(device).to(device)
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

test_img = rearrange(sample_image * 255.0, 'c h w -> h w c')
wandb.log({"examples": [wandb.Image(test_img, caption="example image original")]})
test_img_input = sample_image

wandb.watch(model, mse_loss, log="all", log_graph=True)

for epoch in range(num_epochs):

    model.train()

    pbar = tqdm(range(num_train_batches), desc=f"Epoch {epoch}")

    train_loss = 0

    for i in pbar:
        x, y, t = load_batch(train_data[i])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        t = torch.from_numpy(t)
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
        x = np.concatenate((test_img_input, np.ones(1, w, h)), axis=0)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).to(device=device, dtype=torch.float)

        for i in range(max_time):
            time_input = np.array([i/max_time])
            time_input = torch.from_numpy(time_input).to(device=device, dtype=torch.float)
            test_img_pred = model(x, time_input)
            x = np.concatenate((test_img_input, test_img_pred.detach().cpu().numpy()[0]), axis=0)
            x = np.expand_dims(x, axis=0)
            x = torch.from_numpy(x).to(device=device, dtype=torch.float)
        
        test_img_rearranged = test_img_pred.detach().cpu().numpy()[0] * 255.0
        test_img_rearranged = rearrange(test_img_rearranged, 'c h w -> h w c')
        wandb.log({"outputs": [wandb.Image(test_img_rearranged, caption=f"depth at epoch {epoch}")]})

        val_loss_num = 0

        for i in range(num_val_batches):
            x, y, t = load_batch(val_data[i])
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            t = torch.from_numpy(t)
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