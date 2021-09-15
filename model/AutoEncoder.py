import torch
from torch import nn
from model.InvertedResidual import InvertedResidual


class Encoder(nn.Module):
    def __init__(self, z_dim, image_size, channels):
        super().__init__()

        self.image_size = image_size

        conv_list = []
        conv_list.append(nn.Dropout(0.01))
        for i in range(len(channels)-1):
            conv_list.append(
                InvertedResidual(
                    channels[i],
                    channels[i+1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    expand_ratio=6,
                ))
        conv_list.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_list)

        feature_size = image_size // 2**(len(channels)-1)
        feature_dim = channels[-1] * feature_size ** 2

        self.dense = nn.Linear(feature_dim, z_dim)

    def forward(self, x):
        x = self.conv(x)
        image_feature = self.dense(x)
        return image_feature

class Decoder(nn.Module):
    def __init__(self, z_dim, n_channel):
        super().__init__()

        input_dim = 2 + z_dim

        units = [input_dim, 128, 256, 128, n_channel]
        layer_list = []
        for i in range(0, len(units)-1):
            layer_list.append(nn.Linear(units[i], units[i+1]))
            if i != len(units)-2:
                layer_list.append(nn.ReLU())
        layer_list.append(nn.Sigmoid())
        self.dense = nn.Sequential(*layer_list)

    def forward(self, z, image_size):
        if type(image_size) == int:
            height = width = image_size
        else:
            height, width = image_size
        x = torch.tile(torch.linspace(0, 1, width), dims=(height,))
        y = torch.linspace(0, 1, height).repeat_interleave(width)
        xy = torch.t(torch.stack([x, y])).to(z.device)
        xy = xy.repeat(z.shape[0], 1, 1)
        z = z.repeat(height * width, 1, 1).permute(1, 0, 2)
        z = torch.cat([xy, z], dim=2)
        image = self.dense(z)
        image = image.permute(0, 2, 1)
        image = image.reshape(image.shape[0], image.shape[1], height, width)
        return image


class AutoEncoder(nn.Module):
    def __init__(self, z_dim=2, image_size=64, n_channel=3):
        super().__init__()

        self.image_size = image_size

        channels = [n_channel, 8, 16, 32]
        self.encoder = Encoder(z_dim, image_size, channels)
        self.decoder = Decoder(z_dim, n_channel)

    def forward(self, x):
        image_feature = self.encoder(x)
        image_hat = self.decoder(image_feature, self.image_size)
        return image_hat, image_feature