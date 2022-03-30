import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import List, Optional, Tuple
from collections.abc import Iterable

import re
import numpy as np


def rasterio_open(
        image_path: str,
        window: Optional[Tuple[int, int, int, int]] = None,
        channels: Optional[List[int]] = None,
        fix_nans: bool = False):
    """
    Load an image through rasterio

    :param image_path: Path to the image
    :param window: section of the image to load. It is a tuple (col_off, row_off, width, height)
    :param channels: select the channels to export
    :param fix_nans: check if nan in image and substitute with 0

    :return: loaded image as numpy float array
    """

    assert Path(image_path).exists(), f'Image {image_path} does not exist'

    with rasterio.open(image_path) as f:
        if window:
            col_off, row_off, width, height = window
            assert 0 <= col_off <= f.width - width, f'Window height offset should be between 0 and {f.width} - {width}'
            assert 0 <= row_off <= f.height - height,  f'Window width offset should be between 0 and {f.height} - {height}'
            assert 0 < height, 'Window height should be greater than 0'
            assert 0 < width, 'Window width should be greater than 0'

            image = f.read(window=Window(col_off, row_off, width, height))
        else:
            image = f.read()

        if channels:
            image = image[channels]

        if fix_nans:
            return np.nan_to_num(image, copy=True).astype(np.float32)
        return image.astype(np.float32)


def rasterio_open_multiple(
        image_paths: List[str],
        window: Optional[Tuple[int, int]] = None,
        fix_nans: bool = False):
    """
    Load an image list through rasterio and concat it.
    Useful for tif files with different bands in different files.

    :param image_paths: Paths to the images as list
    :param window: section of the image to load. It is a tuple (col_off, row_off, width, height)
    :param fix_nans: check if nan in image and substitute with 0

    :return: loaded image as numpy float array where all the images have been concatenated on the channel dimension
    """

    images = [rasterio_open(image_path, window, fix_nans) for image_path in image_paths]

    return np.concatenate(images) # TODO check


def rasterio_open_multiple_files(
        image_paths: List[str],
        window: Optional[Tuple[int, int, int, int]] = None,
        channels: Optional[List[int]] = None,
        fix_nans: bool = False):
    """
    Load a list of tif files through rasterio and stack them.

    :param image_paths: Paths to the images as list
    :param window: section of the image to load. It is a tuple (col_off, row_off, width, height)
    :param channels: select the channels to export
    :param fix_nans: check if nan in image and substitute with 0

    :return: loaded image as numpy float array where all the images have been concatenated on the channel dimension
    """

    images = [rasterio_open(image_path, window, channels, fix_nans) for image_path in image_paths]

    return np.stack(images, axis=-1) # TODO check

def rasterio_get_sizes(image_path: str):
    with rasterio.open(image_path) as f:
        return len(f.descriptions), f.width, f.height


def rasterio_get_descriptors(image_path: str):
    with rasterio.open(image_path) as f:
        return f.descriptions


def filter_pathlist(path_list, expr):
    if expr == 'all':
        return path_list

    elif expr[:2] == 'I:':
        fl = eval(f'path_list[{expr[2:]}]')
        if type(fl) == str:
            return [fl]
        return fl

    elif expr[:2] == 'R:':
        regexp = re.compile(expr[2:])
        return [e for e in path_list if regexp.match(e.name)]

    elif isinstance(expr, list):
        not_included = set(path_list) - set(expr)
        assert not not_included, \
                f'Some experiences could not be found: {not_included}'

        return expr

    raise NotImplementedError


def deep_compare(sample_shape1, sample_shape2):
    assert type(sample_shape1) == type(sample_shape2), \
        f'{type(sample_shape1)} != {type(sample_shape2)}'
           

    if sample_shape1 is None or sample_shape2 is None:
        return None

    if isinstance(sample_shape1, Iterable):
        assert len(sample_shape1) == len(sample_shape2)
        return tuple([deep_compare(s1, s2) for s1, s2 in zip(sample_shape1, sample_shape2)])

    return sample_shape1 if sample_shape2 == sample_shape1 else None


class SampleShape:
    def __init__(self, dim, is_windowable: bool = False):
        if isinstance(dim, tuple):
            self.dim = list(dim)
        elif isinstance(dim, SampleShape):
            self.dim = dim.dim
        else:
            self.dim = dim

        self.is_windowable = is_windowable

    def __getitem__(self, idx):
        if self.is_leaf:
            if len(self.dim) <= idx:
                return None
            return self.dim[idx]

        return [d[idx] for d in self.dim]

    def apply_windowing(self, window_size):
        if not self.is_windowable:
            pass
        elif self.is_leaf:
            self.dim[-2] = window_size[0]
            self.dim[-1] = window_size[1]
        else:
            for d in self.dim:
                d.apply_windowing(window_size)

    def insert(self, idx, value):
        if self.is_leaf:
            self.dim.insert(idx, value)
        else:
            for d in self.dim:
                d.insert(idx, value)

    @property
    def is_leaf(self):
        if len(self.dim) == 0:
            return True
        elif isinstance(self.dim[0], int) or self.dim[0] is None:
            return True
        return False

    def __len__(self):
        if self.is_leaf:
            return len(self.dim)
        else:
            return [len(d) for d in self.dim]

    def __eq__(self, other):
        if self.is_leaf and not other.is_leaf:
            return False

        if not self.is_leaf and other.is_leaf:
            return False

        if self.is_leaf and other.is_leaf:
            return self.dim == other.dim

        return all([d1 == d2 for d1, d2 in zip(self.dim, other.dim)])
    
    def merge(self, other):
        if self.is_leaf and not other.is_leaf:
            return SampleShape(())

        if not self.is_leaf and other.is_leaf:
            return SampleShape(())

        if self.is_leaf and other.is_leaf:
            return SampleShape([d1 if d1 == d2 else None for d1, d2 in zip(self.dim, other.dim)])

        return SampleShape([d1.merge(d2) for d1, d2 in zip(self.dim, other.dim)])

    def to_tuple(self):
        if self.is_leaf:
            return tuple(self.dim)
        return tuple(d.to_tuple() for d in self.dim)
