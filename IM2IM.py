import torch
from torch import nn

import sys
sys.path.append('.')


class DepthwiseSeparableConvolutions(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.depthwiseSeparableConvolutions = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.depthwiseSeparableConvolutions(x)


class Encoder(nn.Module):
    def __init__(self, image_feature_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            DepthwiseSeparableConvolutions(3, 32, kernel_size=5, stride=2, padding=2),
            DepthwiseSeparableConvolutions(32, 64, kernel_size=5, stride=2, padding=2),
            DepthwiseSeparableConvolutions(64, 128, kernel_size=5, stride=2, padding=2),
            DepthwiseSeparableConvolutions(128, 256, kernel_size=5, stride=2, padding=2),
            nn.Flatten(),
            nn.Linear(256 * 8 ** 2, 256),
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
            # nn.Linear(image_feature_dim, 1000),
            nn.ReLU(),
            nn.Linear(256, 256 * 8 ** 2),
            # nn.Linear(1000, 128 * 16 ** 2),
            nn.ReLU(),
        )

        self.deconvolution = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),  # 64 -> 128
            nn.Sigmoid(),
        )

    def forward(self, feature):
        x = self.fully_connected(feature)
        batch_size, steps, _ = x.shape
        x = x.reshape(batch_size * steps, 256, 8, 8)
        # x = x.reshape(batch_size * steps, 128, 16, 16)
        y = self.deconvolution(x)
        return y


class IM2IM(nn.Module):
    def __init__(self, state_dim=9, image_feature_dim=15, LSTM_dim=100):
        super().__init__()

        self.encoder = Encoder(image_feature_dim)
        self.decoder = Decoder(image_feature_dim)

        self.state_dim = state_dim
        self.image_feature_dim = image_feature_dim

        self.lstm = nn.LSTM(image_feature_dim + state_dim, LSTM_dim,
                            num_layers=2, batch_first=True)
        self.linear = nn.Linear(LSTM_dim - image_feature_dim, state_dim)

    def forward(self, state, image):
        batch_size, steps, channel, imsize, _ = image.shape
        image = image.reshape(batch_size * steps, channel, imsize, imsize)

        image_feature = self.encoder(image)
        image_feature = image_feature.reshape(batch_size, steps, -1)

        x = torch.cat([image_feature, state], axis=2)
        y, (h, c) = self.lstm(x)
        image_feature_hat = y[:, :, :self.image_feature_dim]
        state_feature = y[:, :, self.image_feature_dim:]
        state_hat = self.linear(state_feature)

        image_hat = self.decoder(image_feature_hat)

        image_hat = image_hat.reshape(batch_size, steps, channel, imsize, imsize)
        state_hat = state_hat.reshape(batch_size, steps, -1)
        return state_hat, image_hat


class IM2IMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, state, state_hat, image, image_hat):
        loss_state = self.mse(state, state_hat)
        loss_image = self.mse(image, image_hat)
        loss = loss_state + loss_image
        return loss, loss_state, loss_image