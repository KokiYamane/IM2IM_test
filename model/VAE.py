import torch
from torch import nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, expand_ratio=6):
        super().__init__()

        hidden_channels = round(in_channels * expand_ratio)

        self.invertedResidual = nn.Sequential(
            # point wise
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(),

            # depth wise
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(),

            # point wise
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.invertedResidual(x)


class Encoder(nn.Module):
    def __init__(self, z_dim, image_size, channels, label_dim=0):
        super().__init__()

        self.image_size = image_size

        if label_dim != 0:
            label_vec_dim = 2
            self.dense_label = nn.Linear(label_dim, label_vec_dim)
            import copy
            channels = copy.deepcopy(channels)
            channels[0] += label_vec_dim

        conv_list = []
        conv_list.append(nn.Dropout(0.01))
        for i in range(len(channels)-1):
            # conv_list.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2))
            conv_list.append(InvertedResidual(channels[i], channels[i+1], kernel_size=5, stride=2, padding=2, expand_ratio=6))
        conv_list.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_list)

        feature_size = image_size // 2**(len(channels)-1)
        feature_dim = channels[-1] * feature_size ** 2

        self.dense_encmean = nn.Linear(feature_dim, z_dim)
        self.dense_encvar = nn.Linear(feature_dim, z_dim)

    def forward(self, x, label=None):
        if label != None:
            if label.dim() == 1:
                label = label.unsqueeze(1)
            v = self.dense_label(label.float())
            v = v.repeat(self.image_size, self.image_size, 1, 1)
            v = v.reshape(x.shape[0], -1, self.image_size, self.image_size)
            x = torch.cat([x, v], dim=1)
        x = self.conv(x)
        mean = self.dense_encmean(x)
        std = F.relu(self.dense_encvar(x))
        return mean, std


class Decoder(nn.Module):
    def __init__(self, z_dim, n_channel, label_dim=0):
        super().__init__()

        input_dim = 2 + z_dim
        if label_dim != 0:
            label_vec_dim = 2
            self.dense_label = nn.Linear(label_dim, label_vec_dim)
            input_dim += label_vec_dim

        units = [input_dim, 128, 256, 128, n_channel]
        layer_list = []
        for i in range(0, len(units)-1):
            layer_list.append(nn.Linear(units[i], units[i+1]))
            if i != len(units)-2:
                layer_list.append(nn.ReLU())
        layer_list.append(nn.Sigmoid())
        self.dense = nn.Sequential(*layer_list)

    def forward(self, z, image_size, label=None):
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
        if label != None:
            v = self.dense_label(label.float()).to(z.device)
            v = v.repeat(height * width, 1, 1).permute(1, 0, 2)
            z = torch.cat([z, v], dim=2)
        image = self.dense(z)
        image = image.permute(0, 2, 1)
        image = image.reshape(image.shape[0], image.shape[1], height, width)
        return image


class VAE(nn.Module):
    def __init__(self, z_dim=2, image_size=64, n_channel=3, label_dim=0):
        super().__init__()

        self.image_size = image_size

        channels = [n_channel, 8, 16, 32, 64, 128]
        self.encoder = Encoder(z_dim, image_size, channels, label_dim)
        self.decoder = Decoder(z_dim, n_channel, label_dim)

    def _sample_z(self, mean, std):
        epsilon = torch.randn(mean.shape).to(mean.device)
        return mean + std * epsilon

    def forward(self, x, label=None):
        mean, std = self.encoder(x, label)
        z = self._sample_z(mean, std)
        y = self.decoder(z, self.image_size, label)
        return y, mean, std

    def encode(self, x, label=None):
        mean, std = self.encoder(x, label)
        return mean
