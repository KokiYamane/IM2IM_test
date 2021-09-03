import torch
from torch import nn

import sys
sys.path.append('.')
from model.AutoEncoder import Encoder, Decoder
from model.SpatialAE import SpatialEncoder


# Spatial Attention Point Network
class SPAN(nn.Module):
    def __init__(self, state_dim=9, keypoints_num=16, image_feature_dim=30,
                 LSTM_dim=100, LSTM_layer_num=2):
        super().__init__()

        self.keypoints_num = keypoints_num
        self.state_dim = state_dim
        self.LSTM_dim = LSTM_dim
        self.LSTM_layer_num = LSTM_layer_num

        n_channel = 3
        self.image_size = 64
        self.encoder = Encoder(z_dim=image_feature_dim, image_size=self.image_size,
                               channels=[n_channel, 8, 16, 32])
        self.spatial_encoder = SpatialEncoder(channels=[n_channel, 8, 16, keypoints_num])
        self.decoder = Decoder(z_dim=image_feature_dim, n_channel=n_channel)

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
        image_hat = self.decoder(image_feature, image_size=self.image_size)

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