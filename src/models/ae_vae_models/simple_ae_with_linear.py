from torch import nn, Tensor
from typing import Tuple

from .base_ae import BaseAE


class SimpleAEWithLinear(BaseAE):
    def __init__(self,
                 input_shape: Tuple[int],
                 latent_dim: int,
                 visualisation_channels):
        super().__init__(visualisation_channels)

        self.latent_dim = latent_dim

        channels = input_shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 7),
            nn.LeakyReLU()
        )

        self.width = (input_shape[1] // 2) // 2 - 6
        self.lin_enc = nn.Linear(256 * self.width * self.width, latent_dim)
        self.lin_dec = nn.Linear(latent_dim, 256 * self.width * self.width)

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
        x = self.encoder(input)
        return self.lin_enc(x.view(x.shape[0], -1))

    def decode(self, input: Tensor) -> Tensor:
        x = self.lin_dec(input)
        x = x.view(input.shape[0], 256, self.width, self.width)
        return self.decoder(x)
