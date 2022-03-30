import kornia
from typing import List, Dict
import torch
from src.utils import load_obj


class Normaliser:
    def __init__(self, image_names: List[str], **kwargs):
        self.image_names = image_names

    def __call__(self, input):
        raise NotImplementedError

    def inverse(self, input):
        raise NotImplementedError


class NoNormaliser(Normaliser):
    def __call__(self, input):
        return input

    def inverse(self, input):
        return input


class MeanStdNormaliser:
    def __init__(self, image_names: List[str], mean: float = 0, std: float = 1,
                 **kwargs):
        super().__init__(image_names, **kwargs)
        self.normalisation = kornia.augmentation.Normalize(mean=mean, std=std)
        self.denormaliser = kornia.augmentation.Denormalize(mean=mean, std=std)

    def __call__(self, input):
        return self.normalisation(input)

    def inverse(self, input):
        return self.denormaliser(input)


class LogNormaliser(Normaliser):
    def __call__(self, input):
        input = input.log()
        input[torch.isinf(input)] = float('nan')
        return input

    def inverse(self, input):
        return torch.exp(input)


class MinMaxNormaliser(Normaliser):
    def __init__(self, image_names: List[str], min_clip: float, max_clip: float, y0: float, y1: float, **kwargs):
        super().__init__(image_names, **kwargs)

        self.clip_norm = ClipNormaliser(min_clip, max_clip)
        self.resc_norm = RescaleNormaliser(min_clip, max_clip, y0, y1)

    def __call__(self, input):
        return self.resc_norm(self.clip_norm(input))

    def inverse(self, input):
        return self.clip_norm.inverse(self.resc_norm.inverse(input))


class ClipNormaliser(Normaliser):
    def __init__(self, image_names: List[str], min_clip: float = None,
                 max_clip: float = None, **kwargs):
        super().__init__(image_names, **kwargs)

        self.min_clip = min_clip
        self.max_clip = max_clip

    def __call__(self, input):
        if self.min_clip is not None:
            input[input < self.min_clip] = self.min_clip
        if self.max_clip is not None:
            input[input > self.max_clip] = self.max_clip
        return input

    def inverse(self, input):
        # Cannot properly invert! Lost information...
        return input


class RescaleNormaliser(Normaliser):
    def __init__(self, image_names: List[str], x0: float, x1: float, y0:
                 float, y1: float, **kwargs):
        super().__init__(image_names, **kwargs)

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def __call__(self, input):
        input = ((input - self.x0) / (self.x1 - self.x0)) \
            * (self.y1 - self.y0) + self.y0
        return input

    def inverse(self, input):
        input = ((input - self.y0) * (self.x1 - self.x0)) \
            / (self.y1 - self.y0) + self.x0
        return input


class CompositeNormaliser(Normaliser):
    """
    Normaliser with different transformations applied to each channel
    """

    def __init__(self, image_names, normaliser_specs: Dict[str, Dict],
                 **kwargs):
        super().__init__(image_names, **kwargs)

        self.normalisers = \
            [self.make_normaliser(normaliser_specs.get(im_n, {})) for im_n in image_names]

    def make_normaliser(self, norm_spec: Dict = {}):
        norm_cls = norm_spec.get('normaliser_cls', 'src.data.normalisers.NoNormaliser')
        norm_args = norm_spec.get('normaliser_args', {})

        return load_obj(norm_cls)(self.image_names, **norm_args)

    def __call__(self, input):
        channels = input.shape[0]
        for i in range(channels):
            input[i, :, :] = self.normalisers[i](input[i, :, :])
        return input

    def inverse(self, input):
        channels = input.shape[0]
        for i in range(channels):
            input[i, :, :] = self.normalisers[i].inverse(input[i, :, :])
        return input


class ListNormaliser(Normaliser):
    def __init__(self, image_names: List[str], normaliser_specs: List[Dict],
                 **kwargs):
        super().__init__(image_names, **kwargs)

        self.normalisers = \
            [self.make_normaliser(ns) for ns in normaliser_specs]

    def make_normaliser(self, norm_spec: Dict):
        norm_cls = list(norm_spec.keys())[0]
        norm_args = norm_spec[norm_cls]

        return load_obj(norm_cls)(self.image_names, **norm_args)

    def __call__(self, input):
        for n in self.normalisers:
            input = n(input)
        return input

    def inverse(self, input):
        for n in self.normalisers[::-1]:  # going in reverse order
            input = n.inverse(input)
        return input


if __name__ == '__main__':
    import torch

    input = (torch.randn(1, 8, 8) + 1) * 2
    #image_names = [f"B{i}" for i in range(1)]

    input1 = ClipNormaliser(0, 0.5)(input)
    assert input1.max() <= 0.5, f'max = {input1.max()} should be less than 0.5'
    assert input1.min() >= 0, f'min = {input1.min()} should be greater than 0'

    input2 = RescaleNormaliser(0, 0.5, -1, 1)(input1)
    assert input2.max() == 1, f'max = {input2.max()} should be equal to 1'
    assert input2.min() == -1, f'min = {input2.min()} should be equal to -1'
    input1_inversion = RescaleNormaliser(0, 0.5, -1, 1).invert(input2)
    assert torch.all(input1_inversion == input1)

    input3 = MinMaxNormaliser(0, 0.5, -1, 1)(input)
    assert (input2 == input3).all(), "MinMaxNormaliser should be equal to sequence of Clip and Rescale"

    input = (torch.randn(32, 3, 8, 8) + 1) * 2
    normalisers = [ClipNormaliser(0, 0.5), ClipNormaliser(0, 0.2), ClipNormaliser(-0.1, 0.9)]
    input4 = CompositeNormaliser(normalisers)(input)
