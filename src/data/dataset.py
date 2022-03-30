from pathlib import Path
from typing import List, Dict
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import kornia

from src.utils import load_obj

from .utils import rasterio_open, rasterio_get_sizes, rasterio_get_descriptors, SampleShape
from .tiling_strategy import TilingStrategyDummy
from src.data.filters import NoFilter


class TiledDataset(Dataset):
    def __getitem__(self, idx):
        """
        Slices and returns the idx element of each dataset.

        :param idx: index of the element

        :return: a list of sliced items, one for every dataset, sliced in the
                 same window position
        """

        nosspi = self.tiling_strategy.number_of_spatial_sample_per_image

        window_idx = idx % nosspi
        sample_idx = idx // nosspi

        window = self.tiling_strategy.get_window(window_idx)

        return self._get_datum(sample_idx, window)

    def _get_datum(self, idx, window=None):
        raise NotImplementedError

    @property
    def sample_shape(self):
        raise NotImplementedError


class LocationDataset(TiledDataset):
    def __init__(self,
                 main_folder: str,
                 dataset_specs: List[Dict],
                 tiling_strategy_cls: str = 'src.data.tiling_strategy.TilingStrategyDummy',
                 tiling_strategy_args: Dict = {}):
        """
        Dataset that pairs a list of datasets and sample them together,
        returning the element idx of all the datasets when queried for a
        sample.

        :param datasets: the list of datasets
        """
        super().__init__()

        assert Path(main_folder).exists(), \
            f'Folder {main_folder} does not exist'
        self.main_folder = main_folder

        self.datasets = \
            [self.make_dataset(main_folder, ds) for ds in dataset_specs]

        dataset_len = len(self.datasets[0])
        for d in self.datasets[1:]:
            assert len(d) == dataset_len

        # check all datasets have same dimensions and at least one has shape
        self._sample_shape = SampleShape([deepcopy(d.sample_shape) for d in self.datasets], True)
        image_widths = [w for w in self._sample_shape[-2] if w is not None]
        image_heights = [h for h in self._sample_shape[-1] if h is not None]

        for h, w in zip(image_heights[1:], image_widths[1:]):
            assert w == image_widths[0], \
                f"({h}, {w}) != ({image_heights[0]}, {image_widths[0]})"
            assert h == image_heights[0], \
                f"({h}, {w}) != ({image_heights[0]}, {image_widths[0]})"

        tiling_strategy_cls = load_obj(tiling_strategy_cls)
        self.tiling_strategy = tiling_strategy_cls(
            image_shape=(image_widths[0], image_heights[0]),
            **tiling_strategy_args)

        self._sample_shape.apply_windowing(self.tiling_strategy.window_size)

    @staticmethod
    def make_dataset(root_folder, dataset_specs):
        _dataset_cls = list(dataset_specs.keys())[0]
        dataset_cls = load_obj(_dataset_cls)
        return dataset_cls(root_folder, **dataset_specs[_dataset_cls])

    def _get_datum(self, idx, window=None):
        sample = []
        for d in self.datasets:
            sample.append(d._get_datum(idx, window))
        return sample

    def __len__(self):
        """
        returns the lenght of the dataset, depending on the number of tiles per
        image that wiil be generated

        :return: number of elements in the dataset
        """
        return len(self.datasets[0]) * \
            self.tiling_strategy.number_of_spatial_sample_per_image

    @property
    def sample_shape(self):
        return self._sample_shape


class SingleFolderImageDataset(TiledDataset):
    is_image_dataset = True

    def __init__(self,
                 root_folder: str,
                 folder_path: str = '.',
                 check_sizes: bool = False,
                 channels: List[str] = None,
                 filter_cls: str = 'src.data.filters.NoFilter',
                 filter_args: Dict = {},
                 normaliser_cls: str = 'src.data.normalisers.NoNormaliser',
                 normaliser_args: Dict = {},
                 tiling_strategy_cls: str = 'src.data.tiling_strategy.TilingStrategyDummy',
                 tiling_strategy_args: Dict = {}):

        super().__init__()
        
        self.tifs = load_obj(filter_cls)(root_folder, folder_path, '*.tif',
                                         **filter_args)()
        if not self.tifs:
            print(f'Dataset in {root_folder}/{folder_path} turns out to be empty')
            return

        image_channels, image_width, image_height = \
            self.extract_and_check_image_sizes(self.tifs, check_sizes)
        self.image_names = self.extract_image_names(self.tifs)

        if channels:
            self.channels = [int(self.image_names.index(c)) for c in channels]
            self.image_names = channels
        else:
            self.channels = list(range(len(self.image_names)))

        self.image_channels = len(self.channels)

        self.normaliser = \
            load_obj(normaliser_cls)(self.image_names, **normaliser_args)

        tiling_strategy_cls = load_obj(tiling_strategy_cls)
        self.tiling_strategy = tiling_strategy_cls(
            image_shape=(image_width, image_height),
            **tiling_strategy_args)

        self._sample_shape = SampleShape((self.image_channels, image_width, image_height), True)
        self._sample_shape.apply_windowing(self.tiling_strategy.window_size)

    @staticmethod
    def extract_image_names(tifs):
        return rasterio_get_descriptors(tifs[0])

    @staticmethod
    def extract_and_check_image_sizes(tifs, check_sizes=False):
        image_channels, image_width, image_height = rasterio_get_sizes(tifs[0])

        for tif in tifs[1:]:
            channels, width, height = rasterio_get_sizes(tif)
            if check_sizes:
                assert image_width == width, \
                        f'{tif} has width {width} != {image_width}'
                assert image_height == height, \
                        f'{tif} has width {height} != {image_height}'
                assert image_channels == channels, \
                        f'{tif} has width {channels} != {image_channels}'

            image_channels = min(image_channels, channels)
            image_width = min(image_width, width)
            image_height = min(image_height, height)

        return (image_channels, image_width, image_height)

    @property
    def sample_shape(self):
        return self._sample_shape

    def _read_datum(self, idx, window=None):
        image = \
            rasterio_open(self.tifs[idx], window, channels=self.channels)  # ch, h, w
        image = kornia.utils.image_to_tensor(image, keepdim=True)  # w, ch, h
        image = image.permute(1, 2, 0)  # 1, 2, 0 => ch, h, w
        return image

    def _get_datum(self, idx, window=None):
        image = self._read_datum(idx, window)
        image = self.normaliser(image)
        return image

    def __len__(self):
        return len(self.tifs) * \
            self.tiling_strategy.number_of_spatial_sample_per_image


class SingleFolderChangeDataset(TiledDataset):
    is_image_dataset = True

    def __init__(self,
                 root_folder: str,
                 folder_path: str = '.',
                 folder_path_s2: str = '.',
                 filter_cls: str = 'src.data.filters.NoFilter',
                 filter_args: Dict = {},
                 tiling_strategy_cls: str = 'src.data.tiling_strategy.TilingStrategyDummy',
                 tiling_strategy_args: Dict = {}):

        super().__init__()

        s2_tifs = load_obj(filter_cls)(root_folder, folder_path_s2, '*.tif',
                                       **filter_args)()
        if not s2_tifs:
            print(f'Dataset in {root_folder}/{folder_path} turns out to be empty')
            return
        self.fake_len = len(s2_tifs)

        tifs = NoFilter(root_folder, folder_path, '*.tif')()
        if not tifs:
            print(f'Dataset in {root_folder}/{folder_path} turns out to be empty')
            return
        self.tif = tifs[0]
        self.image_channels, self.image_width, self.image_height = rasterio_get_sizes(self.tif)
        self.channels = list(range(self.image_channels))

        self.single_index = self.find_the_index(self.tif, s2_tifs)

        tiling_strategy_cls = load_obj(tiling_strategy_cls)
        self.tiling_strategy = tiling_strategy_cls(
            image_shape=(self.image_width, self.image_height),
            **tiling_strategy_args)

        self._sample_shape = SampleShape((self.image_channels, self.image_width, self.image_height), True)
        self._sample_shape.apply_windowing(self.tiling_strategy.window_size)

    @staticmethod
    def find_the_index(tif_to_find, tifs):
        return [t.name for t in tifs].index(tif_to_find.name)

    @property
    def sample_shape(self):
        return self._sample_shape

    def _read_datum(self, idx, window=None):
        if idx != self.single_index:
            if not window:
                h, w = self.image_height, self.image_width
            else:
                w, h = window[2], window[3]
            return torch.ones(self.image_channels, h, w) * 2

        image = \
            rasterio_open(self.tif, window, channels=self.channels)  # ch, h, w
        image = kornia.utils.image_to_tensor(image, keepdim=True)  # w, ch, h
        image = image.permute(1, 2, 0)  # 1, 2, 0 => ch, h, w
        return image

    def _get_datum(self, idx, window=None):
        image = self._read_datum(idx, window)
        return image

    def __len__(self):
        return self.fake_len * \
            self.tiling_strategy.number_of_spatial_sample_per_image


class NConsecutiveDataset(TiledDataset):
    def __init__(self,
                 main_folder: str,
                 dataset_cls: str,
                 dataset_cls_args: Dict,
                 sequence_length: int,
                 tiling_strategy_cls: str,
                 tiling_strategy_args: Dict):
        super().__init__()

        self.dataset = load_obj(dataset_cls)(main_folder, **dataset_cls_args)
        self.dataset.tiling_strategy = TilingStrategyDummy()

        self._sample_shape = SampleShape(self.dataset.sample_shape, True)
        self._sample_shape.insert(0, sequence_length)
        image_widths = [w for w in self._sample_shape[-2] if w is not None]
        image_heights = [h for h in self._sample_shape[-1] if h is not None]

        for w in image_widths[1:]:
            assert w == image_widths[0]
        for h in image_heights[1:]:
            assert h == image_heights[0]

        tiling_strategy_cls = load_obj(tiling_strategy_cls)
        self.tiling_strategy = tiling_strategy_cls(
            image_shape=(image_widths[0], image_heights[0]),
            **tiling_strategy_args)

        self._sample_shape.apply_windowing(self.tiling_strategy.window_size)

        self.sequence_length = sequence_length

    def _get_datum(self, idx, window=None):
        range_i = range(idx, idx + self.sequence_length)
        return [self.dataset._get_datum(i, window) for i in range_i]

    def __len__(self):
        return (len(self.dataset) - self.sequence_length + 1) * \
            self.tiling_strategy.number_of_spatial_sample_per_image

    @property
    def sample_shape(self):
        return self._sample_shape
