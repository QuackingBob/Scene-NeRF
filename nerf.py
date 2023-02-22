import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# function that takes two images x and y and returns the mean squared error between them
img2mse = lambda x, y : torch.mean((x-y)**2)
# function that takes the mean squared error value x and returns the peak signal-to-noise ratio (PSNR) between the two images
mse2psnr = lambda x : 10 * torch.log10(255**2 / x)
# function that scales and clips the input image x between 0 and 1 and converts it to 8-bit unsigned integer format
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Positional Encodings
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_e

class NeRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward():
        pass
