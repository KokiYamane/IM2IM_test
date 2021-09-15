import torch
from torch import nn
import sys
sys.path.append('.')
from model.AutoEncoder import Encoder, Decoder


class IM2IM(nn.Module):
    def __init__(self, state_dim=9, image_feature_dim=15,
                 LSTM_dim=100, LSTM_layer_num=2):
        super().__init__()

        self.image_size = 64

        self.encoder = Encoder(
            z_dim=image_feature_dim,
            image_size=self.image_size,
            channels=[3, 8, 16, 32],
        )
        self.decoder = Decoder(
            z_dim=image_feature_dim,
            n_channel=3,
        )

        self.state_dim = state_dim
        self.image_feature_dim = image_feature_dim
        self.LSTM_dim = LSTM_dim
        self.LSTM_layer_num = LSTM_layer_num

        self.lstm = nn.LSTM(
            image_feature_dim + state_dim,
            LSTM_dim,
            batch_first=True,
            num_layers=LSTM_layer_num,
            dropout=0.1,
        )
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

        image_feature_hat = image_feature_hat.reshape(batch_size * steps, -1)
        image_hat = self.decoder(image_feature_hat, self.image_size)

        image_hat = image_hat.reshape(
            batch_size, steps, channel, imsize, imsize)
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