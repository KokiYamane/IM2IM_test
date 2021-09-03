import torch
from torch import nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(self, image_feature):
        batch_size, channel, height, width = image_feature.shape
        image_feature = image_feature.reshape(-1, height * width)
        softmax_attention = F.softmax(image_feature / self.temperature, dim=1)
        softmax_attention = softmax_attention.reshape(batch_size, channel, height, width)

        sum_x = torch.sum(softmax_attention, dim=2)
        sum_y = torch.sum(softmax_attention, dim=3)
        device = sum_x.device
        pos_x = torch.linspace(-1, 1, width).tile(batch_size, channel, 1).to(device)
        pos_y = torch.linspace(-1, 1, height).tile(batch_size, channel, 1).to(device)
        feature_points_x = torch.sum(sum_x * pos_x, dim=2)
        feature_points_y = torch.sum(sum_y * pos_y, dim=2)
        return torch.cat([feature_points_x, feature_points_y], dim=1)
