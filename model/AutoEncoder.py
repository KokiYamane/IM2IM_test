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