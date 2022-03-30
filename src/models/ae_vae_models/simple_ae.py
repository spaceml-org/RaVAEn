from torch import nn, Tensor
from typing import Tuple

from .base_ae import BaseAE


class SimpleAE(BaseAE):
    def __init__(self,
                 input_shape: Tuple[int],
                 visualisation_channels):
        super().__init__(visualisation_channels)

        channels = input_shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 7),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=2, padding=1,
                               output_padding=1),
        )

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def decode(self, input: Tensor) -> Tensor:
        result = self.decoder(input)
        return result
