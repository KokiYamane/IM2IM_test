import torch
from torch import nn

import sys
sys.path.append('.')
sys.path.append('..')
from model.InvertedResidual import InvertedResidual
from model.SpatialSoftmax import SpatialSoftmax
from model.AutoEncoder import Decoder


class SpatialEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        conv_list = []
        conv_list.append(nn.Dropout(0.01))
        for i in range(len(channels)-1):
            conv_list.append(
                InvertedResidual(
                    channels[i],
                    channels[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    expand_ratio=6,
                ))
        self.conv = nn.Sequential(*conv_list)

        self.spatial_softmax = SpatialSoftmax()

    def forward(self, x):
        self.feature_map = self.conv(x)
        self.feature_points = self.spatial_softmax(self.feature_map)
        return self.feature_points


class SpatialAE(nn.Module):
    def __init__(self, feature_point_num=16, image_size=64, n_channel=3):
        super().__init__()

        self.image_size = image_size

        channels = [n_channel, 8, 16, feature_point_num]
        self.encoder = SpatialEncoder(channels)
        self.decoder = Decoder(2 * feature_point_num, n_channel)

    def forward(self, x):
        image_feature = self.encoder(x)
        image_hat = self.decoder(image_feature, self.image_size)
        return image_hat, image_feature
