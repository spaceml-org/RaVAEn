from typing import List, Any, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .base_ae import BaseAE

class DeeperAE(BaseAE):

    def __init__(self,
                 input_shape: Tuple[int],
                 hidden_channels: List[int],
                 latent_dim: int,
                 extra_depth_on_scale: int,
                 visualisation_channels,
                 **kwargs) -> None:
        super().__init__(visualisation_channels)
        
        assert input_shape[1]>=2**len(hidden_channels), "Cannot have so many downscaling layers"
        
        self.latent_dim = latent_dim
        print("\nLATENT SPACE size:", latent_dim)
        
        # Calculate size of encoder output
        encoder_output_width = int(input_shape[1]/(2**len(hidden_channels)))
        encoder_output_dim = int(encoder_output_width**2 * hidden_channels[-1])
        self.encoder_output_shape = (hidden_channels[-1], encoder_output_width, encoder_output_width)
        
        if encoder_output_dim<latent_dim:
            raise UserWarning(
                f"Encoder output dim {encoder_output_dim} is smaller than latent dim {latent_dim}."+
                "This means the bottle neck is tighter than intended."
            )
        
        in_channels = input_shape[0]
        
        self.encoder = self._build_encoder([in_channels]+hidden_channels, extra_depth_on_scale)
        
        self.encoder_output = nn.Linear(encoder_output_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, encoder_output_dim)

        self.decoder = self._build_decoder(hidden_channels[::-1]+[in_channels], extra_depth_on_scale)
        
    @staticmethod
    def _build_encoder(channels, extra_depth):
        in_channels = channels[0]
        encoder = []
        for out_channels in channels[1:]:
            encoder+=[
                DownConv(
                    in_channels, 
                    out_channels, 
                    activation=nn.LeakyReLU, 
                    batchnorm=True
                )
            ]
            if extra_depth>0:
                encoder+=[
                    ResConvBlock(
                        out_channels, 
                        out_channels, 
                        activation=nn.LeakyReLU, 
                        batchnorm=True, 
                        depth=extra_depth
                    )
                ]
            # for next time round loop
            in_channels = out_channels
        
        return nn.Sequential(*encoder)
    
    @staticmethod
    def _build_decoder(channels, extra_depth):
        in_channels = channels[0]
        decoder = []
        up_activation = nn.LeakyReLU
        res_activation = nn.LeakyReLU
        for i, out_channels in enumerate(channels[1:]):
            # if last layer use linear activation
            is_last_layer = (i==(len(channels)-2))
            if is_last_layer:
                up_activation = None
            
            decoder+=[
                UpConv(
                    in_channels, 
                    out_channels, 
                    upsample_method='nearest', 
                    activation=up_activation, 
                    batchnorm=not is_last_layer,
                )
            ]
            if extra_depth>0:
                decoder+=[
                    ResConvBlock(
                        out_channels, 
                        out_channels, 
                        activation=res_activation, 
                        batchnorm=not is_last_layer, 
                        depth=extra_depth
                    )
                ]
            # for next time round loop
            in_channels = out_channels
        
        return nn.Sequential(*decoder)

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        latent = self.encoder_output(result)

        return latent

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.encoder_output_shape)
        result = self.decoder(result)
        return result



class ConvBlock(nn.Module):
    """
    Convolutional block which preserves the height and width of the input image.
    
    (convolution => [BN] => LeakyReLU) * depth
    """

    def __init__(self, in_channels, out_channels, depth=2, activation=nn.LeakyReLU, batchnorm=True):
        super().__init__()
        
        layers = []
        for n in range(1, depth+1):                
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            if batchnorm: 
                layers += [nn.BatchNorm2d(out_channels)]
            if activation is not None:
                layers +=[activation()]
            in_channels = out_channels
            
        self.conv_block = nn.Sequential(*layers)


    def forward(self, x):
        return self.conv_block(x)
    


class ResConvBlock(ConvBlock):
    def forward(self, x):
        dx = self.conv_block(x)
        return  x + dx


class DownConv(nn.Module):
    """Downscaling block"""
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU, batchnorm=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]
        if batchnorm: 
            layers += [nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers +=[activation()]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """Upscaling layer with single convolution"""

    def __init__(self, in_channels, out_channels, upsample_method='nearest', 
                 activation=nn.LeakyReLU, batchnorm=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if upsample_method in ['nearest', 'linear', 'bilinear', 'bicubic']:
            align_corners=None if upsample_method=="nearest" else True
            self.up = nn.Sequential(
                nn.Upsample(
                    scale_factor=2, 
                    mode=upsample_method, 
                    align_corners=align_corners
                ),
                ConvBlock(in_channels, out_channels, depth=1, activation=activation, batchnorm=batchnorm)
            )
            # add the single convolution above to keep things even between the number of convolutions
            # in upsamplking and downsampling, regardless of upsampling method
            

        elif upsample_method=='transpose':
            layers = [
                nn.ConvTranspose2d(
                    in_channels, out_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    output_padding=1
                ),
            ]
            if batchnorm: 
                layers += [nn.BatchNorm2d(out_channels)]
            if activation is not None:
                layers +=[activation()]
            self.up = nn.Sequential(*layers)
                
        else:
            raise NotImplementedError(
                f"Upsample method has not been implemented: {upsample_method}"
            )

    def forward(self, x):
        return self.up(x)
