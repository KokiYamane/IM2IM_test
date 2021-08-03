import torch
from torch import nn

import sys
sys.path.append('.')
sys.path.append('..')
# from model.AutoEncoder import Encoder, Decoder
from model.SpatialSoftmax import SpatialSoftmax




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
    def __init__(self, image_feature_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.01),
            InvertedResidual(3, 32, kernel_size=3, stride=1, padding=0, expand_ratio=6),  # 64 -> 62
            # InvertedResidual(16, 32, kernel_size=5, stride=2, padding=0, expand_ratio=6),  # 128 -> 62
            InvertedResidual(32, 64, kernel_size=5, stride=2, padding=0, expand_ratio=6),  # 62 -> 29
            InvertedResidual(64, 128, kernel_size=5, stride=2, padding=0, expand_ratio=6),  # 29 -> 13
            InvertedResidual(128, 256, kernel_size=5, stride=2, padding=0, expand_ratio=6),  # 13 -> 5
            InvertedResidual(256, 512, kernel_size=5, stride=2, padding=0, expand_ratio=6),  # 5 -> 1
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, image_feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, image_feature_dim):
        super().__init__()

        self.fully_connected = nn.Sequential(
            nn.Linear(image_feature_dim, 256),
            nn.ReLU(),
            # nn.Linear(256, 256 * 8 ** 2),
            nn.Linear(256, 128 * 8 ** 2),
            nn.ReLU(),
        )

        self.deconvolution = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),  # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, feature):
        x = self.fully_connected(feature)
        batch_size, _ = x.shape
        # x = x.reshape(batch_size, 256, 8, 8)
        x = x.reshape(batch_size, 128, 8, 8)
        y = self.deconvolution(x)
        return y
class SpatialEncoder(nn.Module):
    def __init__(self, keypoints_num=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, keypoints_num, kernel_size=5, stride=1, padding=2),
            SpatialSoftmax(),
        )

    def forward(self, x):
        return self.encoder(x)


# Spatial Attention Point Network
class SPAN(nn.Module):
    def __init__(self, state_dim=9, keypoints_num=16, image_feature_dim=30,
                 LSTM_dim=100, LSTM_layer_num=2):
        super().__init__()

        self.keypoints_num = keypoints_num
        self.state_dim = state_dim
        self.LSTM_dim = LSTM_dim
        self.LSTM_layer_num = LSTM_layer_num

        self.encoder = Encoder(image_feature_dim=image_feature_dim)
        self.spatial_encoder = SpatialEncoder(
            keypoints_num=self.keypoints_num)
        self.decoder = Decoder(image_feature_dim=image_feature_dim)

        self.heatmap_generation_layer = nn.Linear(
            2 * keypoints_num, image_feature_dim)

        self.lstm = nn.LSTM(2 * keypoints_num + state_dim, LSTM_dim,
            num_layers=LSTM_layer_num, batch_first=True)
        self.linear = nn.Linear(LSTM_dim - 2 * keypoints_num, state_dim)

    def forward(self, state, image, memory=None):
        batch_size, steps, channel, imsize, _ = image.shape
        image = image.reshape(batch_size * steps, channel, imsize, imsize)
        state = state.reshape(batch_size * steps, -1)

        image_feature = self.encoder(image)
        image_feature = image_feature.reshape(batch_size, steps, -1)

        keypoints = self.spatial_encoder(image)

        x = torch.cat([keypoints, state], axis=1)
        x = x.reshape(batch_size, steps, -1)
        y, (h, c) = self.lstm(x, memory)

        keypoints_dim = 2 * self.keypoints_num
        keypoints_hat = y[:, :, :keypoints_dim]
        state_feature = y[:, :, keypoints_dim:]
        state_hat = self.linear(state_feature)
        keypoints = keypoints.reshape(batch_size, steps, -1)
        # print(keypoints.shape)
        keypoints_hat = keypoints_hat.reshape(batch_size, steps, -1)
        # print(keypoints_hat.shape)

        heatmap = self.heatmap_generation_layer(keypoints_hat)
        image_feature = torch.mul(heatmap, image_feature)

        image_feature = image_feature.reshape(batch_size * steps, -1)
        image_hat = self.decoder(image_feature)

        image_hat = image_hat.reshape(batch_size, steps, channel, imsize, imsize)
        state_hat = state_hat.reshape(batch_size, steps, -1)

        if memory == None:
            return state_hat, image_hat, keypoints, keypoints_hat
        else:
            return state_hat, image_hat, keypoints, keypoints_hat, (h, c)


class SPANLoss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, state, state_hat, image, image_hat, keypoints, keypoints_hat):
        loss_state = self.mse(state, state_hat)
        loss_image = self.mse(image, image_hat)
        loss_keypoints = self.alpha * self.mse(keypoints, keypoints_hat)
        loss = loss_state + loss_image + loss_keypoints
        return loss, loss_state, loss_image, loss_keypoints
        