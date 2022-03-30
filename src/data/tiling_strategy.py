from typing import Optional, Tuple

import random


class TilingStrategy:
    def __init__(self,
                 window_size: Tuple[int, int] = None,
                 image_shape: Tuple[int, int] = None,
                 **kwargs):
        """
        Base dataset class for slicing the images belonging to a paired dataset
        when loading.
        It is needed because of rasterio since it can slice the data in
        reading.

        :param datasets: List of paired image-based datasets
        :param window_size: size of the window to open, as (width, height)
        """

        if window_size:
            self.window_size = window_size
        else:
            self.window_size = image_shape
        self.image_width, self.image_height = image_shape

        self.number_of_spatial_sample_per_image = \
            self.get_number_of_spatial_sample_per_image()

    def get_number_of_spatial_sample_per_image(self):
        """
        Depending on the implementation, you can have multiple images resulting
        from a single one.
        This function gives you the number of tiles an image will be
        subsampled.

        :param window_size: size of the window to open, as (width, height)

        :return: number of tiles per image
        """
        raise NotImplementedError

    def get_window(self, idx):
        """
        Get a window for slicing the image.

        :param idx: index of the window

        :return: window, as (col_off, row_off, width, height)
        """
        raise NotImplementedError


class TilingStrategyDummy(TilingStrategy):
    def get_number_of_spatial_sample_per_image(self):
        """
        Depending on the implementation, you can have multiple images resulting
        from a single one.
        This function gives you the number of tiles an image will be
        subsampled.

        :param window_size: size of the window to open, as (width, height)

        :return: number of tiles per image
        """
        return 1

    def get_window(self, idx):
        """
        Get a window for slicing the image.

        :param idx: index of the window

        :return: window, as (col_off, row_off, width, height)
        """
        return None


class TilingStrategyRandomCrop(TilingStrategy):
    def __init__(self,
                 window_size: Tuple[int, int],
                 image_shape: Tuple[int, int],
                 **kwargs):
        """
        Dataset that takes a random crop of the images of the paired datasets
        before
        returning them.

        :param datasets: list of datasets to query
        :param window_size: size of the window to open, as (width, height)
        """
        super().__init__(window_size, image_shape, **kwargs)

        width, height = self.window_size
        self.max_width_sample = self.image_width - width
        self.max_height_sample = self.image_height - height

    def get_number_of_spatial_sample_per_image(self):
        """
        This function gives you the number of tiles an image will be
        subsampled.
        This class will subsample one single tile every time it is called.

        :param window_size: size of the window to open, as (width, height)

        :return: number of tiles per image, 1
        """
        return 1

    def get_window(self, idx):
        """
        Get a random window for slicing the image.

        :param idx: index of the window

        :return: window, as (col_off, row_off, width, height)
        """
        width, height = self.window_size
        col_off = random.randint(0, self.max_width_sample)
        row_off = random.randint(0, self.max_height_sample)

        return (col_off, row_off, width, height)


class TilingStrategyFullGrid(TilingStrategy):
    def __init__(self,
                 window_size: Tuple[int, int],
                 image_shape: Tuple[int, int],
                 overlap: Optional[Tuple[int, int]] = (0, 0),
                 **kwargs):
        """
        Class that samples all the tiles that can be contained in a grid within
        the image itself.

        :param datasets: list of datasets to query
        :param window_size: size of the window to open, as (width, height)
        """
        if isinstance(overlap, str):
            #print("Input:", overlap, type(overlap))
            overlap = overlap.replace("$","") # loads as string of ~ "[0, 0]$" 
            overlap = eval(overlap)
        self.overlap = overlap
        ####print("Evaluated overlap as:", overlap, type(overlap))
        assert 0 <= overlap[0] < window_size[0]
        assert 0 <= overlap[1] < window_size[1]

        super().__init__(window_size, image_shape, **kwargs)

        self.grid_shape = self.get_grid_shape()

    def get_number_of_spatial_sample_per_image(self):
        """
        This function gives you the number of tiles an image will be subsampled
        into.
        In this case, it is the whole set of tiles in the grid we can compose
        using the image.

        :param window_size: size of the window to open, as (width, height)

        :return: number of tiles per image
        """
        grid_shape = self.get_grid_shape()
        return grid_shape[0] * grid_shape[1]

    def get_grid_shape(self):
        """
        function to get the shape of the grid containing the tile windows.

        :param window_size: size of the window to open, as (width, height)

        :return: number of columns and rows of the grid, as (columns, rows)
        """

        horizontal_stride = self.window_size[0] - self.overlap[0]
        horizontal_tiles = \
            (self.image_width - self.window_size[0]) // horizontal_stride + 1

        vertical_stride = self.window_size[1] - self.overlap[1]
        vertical_tiles = \
            (self.image_height - self.window_size[1]) // vertical_stride + 1

        return horizontal_tiles, vertical_tiles

    def get_window(self, idx):
        """
        Get a window for slicing the image.

        :param idx: index of the window

        :return: window, as (col_off, row_off, width, height)
        """
        width, height = self.window_size
        horizontal_stride = self.window_size[0] - self.overlap[0]
        vertical_stride = self.window_size[1] - self.overlap[1]

        col_off = (idx % self.grid_shape[0]) * horizontal_stride
        row_off = (idx // self.grid_shape[0]) * vertical_stride
        return (col_off, row_off, width, height)
