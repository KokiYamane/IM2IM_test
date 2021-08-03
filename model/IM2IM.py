import torch
from torch import nn

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
            InvertedResidual(3, 16, kernel_size=5, stride=2, padding=0, expand_ratio=4),  # 128 -> 62
            InvertedResidual(16, 32, kernel_size=5, stride=2, padding=0, expand_ratio=4),  # 62 -> 29
            InvertedResidual(32, 64, kernel_size=5, stride=2, padding=0, expand_ratio=4),  # 29 -> 13
            nn.Flatten(),
            nn.Linear(64 * 13 ** 2, 256),
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
            nn.Linear(256, 256 * 8 ** 2),
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
        y = self.deconvolution(x)
        return y


class IM2IM(nn.Module):
    def __init__(self, state_dim=9, image_feature_dim=15,
                 LSTM_dim=100, LSTM_layer_num=2):
        super().__init__()

        self.encoder = Encoder(image_feature_dim)
        self.decoder = Decoder(image_feature_dim)

        self.state_dim = state_dim
        self.image_feature_dim = image_feature_dim
        self.LSTM_dim = LSTM_dim
        self.LSTM_layer_num = LSTM_layer_num

        self.lstm = nn.LSTM(image_feature_dim + state_dim, LSTM_dim,
                            num_layers=LSTM_layer_num, batch_first=True)
        self.linear = nn.Linear(LSTM_dim - image_feature_dim, state_dim)

    def forward(self, state, image, memory=None):
        batch_size, steps, channel, imsize, _ = image.shape
        image = image.reshape(batch_size * steps, channel, imsize, imsize)

        image_feature = self.encoder(image)
        image_feature = image_feature.reshape(batch_size, steps, -1)

        x = torch.cat([image_feature, state], axis=2)
        y, (h, c) = self.lstm(x, memory)
        image_feature_hat = y[:, :, :self.image_feature_dim]
        state_feature = y[:, :, self.image_feature_dim:]
        state_hat = self.linear(state_feature)

        image_hat = self.decoder(image_feature_hat)

        image_hat = image_hat.reshape(batch_size, steps, channel, imsize, imsize)
        state_hat = state_hat.reshape(batch_size, steps, -1)

        if memory == None:
            return state_hat, image_hat
        else:
            return state_hat, image_hat, (h, c)


class IM2IMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, state, state_hat, image, image_hat):
        loss_state = self.mse(state, state_hat)
        loss_image = self.mse(image, image_hat)
        loss = loss_state + loss_image
        return loss, loss_state, loss_image
        