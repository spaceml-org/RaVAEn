from typing import List, Dict

import torch

from src.utils import load_obj

class RandomBandShifts(torch.nn.Module):
    """
    Args:
    - p : p probability of jittering the tile (shifted bands), (1-p) of applying a random crop (all bands aligned)
    """
    def __init__(self, tile_size: int, crop_size: int, p: float):
        super().__init__()
        
        assert tile_size > crop_size, "tile must be larger than the desidered crop size"
        assert 0 <= p <= 1, "probability of jittering should be between 0 and 1"

        self.shift = tile_size - crop_size
        self.p = torch.tensor(p)
        self.crop_size = crop_size

        self.center_crop = CenterCrop(crop_size)
 
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        num_bands = x.shape[-3] # (..., C, H, W)
        if torch.bernoulli(self.p) == 1:
            
            # for each band (apart from first), draw randomly the indices of the top left corner of the crop
            indices = torch.randint(high=self.shift+1, size=(num_bands-1, 2))
            
            # center crop first band (keep it as reference)
            x_out = [self.center_crop(x[..., 0, :, :].unsqueeze(-3))]
            
            # TODO: avoid loop by slicing?
            for b in range(num_bands-1):
                x_out.append(x[..., b+1, indices[b, 0]: indices[b, 0] + self.crop_size, indices[b, 1]: indices[b, 1] + self.crop_size].unsqueeze(-3))

            x_out = torch.cat(x_out, dim=-3)

        else:
            # center crop
            x_out = self.center_crop(x)

        return x_out

class CenterCrop(torch.nn.Module):
    
    def __init__(self, crop_size: int):
        super().__init__()
    
        self.crop_size = crop_size
 
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        center = torch.tensor([x.shape[-1] // 2, x.shape[-2] // 2])
        top_left = center - self.crop_size // 2
        x_out = x[..., top_left[0]: top_left[0] + self.crop_size, top_left[1]: top_left[1] + self.crop_size]
        
        return x_out

class Transformer(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()

        transforms = []
        for t in args:
            for t_cls, t_args in t.items():
                transforms.append(load_obj(t_cls)(**t_args["transf_args"]))
                           
        self.transforms = torch.nn.Sequential(*transforms)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)
        return x_out

class NoTransformer(torch.nn.Module):

    def __init__(self, *args):
        super().__init__()        
 
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

if __name__ == "__main__":

    x = torch.arange(100).reshape((4, 5, 5))

    rnd_crop = RandomBandShifts(5, 4, 1.)
    
    assert rnd_crop(x).shape == (4, 4, 4)
    print(rnd_crop(x), x)


    rnd_crop = RandomBandShifts(5, 4, 0.)
    
    assert rnd_crop(x.float()).shape == (4, 4, 4)
    print(rnd_crop(x.float()), x)
