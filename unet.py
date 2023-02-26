import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, device) -> None:
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding="same").to(device)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), padding="same").to(device)
        self.relu = nn.ReLU().to(device)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, device, channels=[3,64,128,256,512,1024]) -> None:
        super(Encoder, self).__init__()
        self.encoding_blocks = nn.ModuleList([Block(channels[i], channels[i+1], device).to(device) for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(2).to(device)
    
    
    def forward(self, x):
        features = []
        for block in self.encoding_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, device, channels=[1024, 512, 256, 128, 64]) -> None:
        super(Decoder, self).__init__()
        self.channels = channels
        self.deconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=(3,3), stride=2, padding=1).to(device) for i in range(len(channels) - 1)])
        self.deconv_blocks = nn.ModuleList([Block(channels[i+1], channels[i+1], device).to(device) for i in range(len(channels) - 1)])
    

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            b, c, h, w = x.shape
            x = self.deconvs[i](x, output_size=(b, self.channels[i+1], 2*h, 2*w))
            x = encoder_features[i] + x
            x = self.deconv_blocks[i](x)
        return x


class Unet(nn.Module):
    def __init__(self, device, inner_channels=[3,64,128,256,512,1024]) -> None:
        super(Unet, self).__init__()
        self.device = device
        self.encoder_channels = inner_channels
        self.decoder_channels = inner_channels[1:][::-1]
        print(self.encoder_channels)
        print(self.decoder_channels)
        self.encoder = Encoder(device, self.encoder_channels).to(device)
        self.decoder = Decoder(device, self.decoder_channels).to(device)
        self.convoutput = nn.Conv2d(in_channels=inner_channels[1], out_channels=1, kernel_size=(1,1), padding="same").to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        

    def forward(self, x):
        encoder_filters = self.encoder(x)
        x = self.decoder(encoder_filters[::-1][0], encoder_filters[::-1][1:])
        x = self.convoutput(x)
        if not self.training:
            x = self.sigmoid(x)
        return x