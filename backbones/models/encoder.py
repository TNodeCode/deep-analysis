import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            self.downblock(3, 16),
            self.downblock(16, 32),
            self.downblock(32, 64),
            self.downblock(64, 128),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

    def downblock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
