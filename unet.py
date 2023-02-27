import torch
import torch.nn as nn
from einops import rearrange


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, device) -> None:
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding="same").to(device)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), padding="same").to(device)
        self.embed = SinusoidalPosEmb(out_channels).to(device)
        self.relu = nn.ReLU().to(device)

    
    def forward(self, x, time):
        h, w = x.shape[2], x.shape[3]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        t = self.embed(time)
        t = t.unsqueeze(-1).unsqueeze(-1)
        # x = torch.cat([x, t.expand(-1, -1, h, w)], dim=1)
        return x + t


class Encoder(nn.Module):
    def __init__(self, device, channels=[4,64,128,256,512,1024]) -> None:
        super(Encoder, self).__init__()
        self.encoding_blocks = nn.ModuleList([Block(channels[i], channels[i+1], device).to(device) for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2).to(device)
    
    
    def forward(self, x, time):
        features = []
        for block in self.encoding_blocks:
            x = block(x, time)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, device, channels=[1024, 512, 256, 128, 64]) -> None:
        super(Decoder, self).__init__()
        self.channels = channels
        self.deconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=(3,3), stride=2, padding=1).to(device) for i in range(len(channels) - 1)])
        self.deconv_blocks = nn.ModuleList([Block(channels[i+1], channels[i+1], device).to(device) for i in range(len(channels) - 1)])
    

    def forward(self, x, encoder_features, time):
        for i in range(len(self.channels) - 1):
            b, c, h, w = x.shape
            x = self.deconvs[i](x, output_size=(b, self.channels[i+1], 2*h, 2*w))
            x = encoder_features[i] + x
            x = self.deconv_blocks[i](x, time)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        dtype = time.dtype
        half_dim = self.dim//2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)
        emb = time.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class Unet(nn.Module):
    def __init__(self, device, inner_channels=[4,64,128,256,512,1024]) -> None:
        super(Unet, self).__init__()
        self.device = device
        self.encoder_channels = inner_channels
        self.decoder_channels = inner_channels[1:][::-1]
        print(self.encoder_channels)
        print(self.decoder_channels)
        self.encoder = Encoder(device, self.encoder_channels).to(device)
        self.decoder = Decoder(device, self.decoder_channels).to(device)
        self.convoutput = nn.Conv2d(in_channels=inner_channels[1], out_channels=1, kernel_size=(1,1), padding="same").to(device)
        self.relu = nn.ReLU().to(device)
        

    def forward(self, x, t):
        encoder_filters = self.encoder(x, t)
        x = self.decoder(encoder_filters[::-1][0], encoder_filters[::-1][1:], t)
        x = self.convoutput(x)
        x = self.relu(x)
        return x