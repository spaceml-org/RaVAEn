# Dataset and datamodule

## Loading a tif image
The main corpus of the datasets is stored as `.tif` images.
To read them, we employ the `rasterio` package, which permits to load an image
through a window query, so that there is no need for the entire image to be
loaded in memory.

The functions responsible for this are located in the `src/data/utils.py`
script.
The main function is `rasterio_open`, which can retrieve and load from a
specific image a window and selected channels.

The window is a tuple of the form `(column, row, width, height)` and is normally
provided by a tiling strategy.

## Tiling strategies
Tiling strategies are collected in the `src/data/tiling_strategy.py` script.
The main purtpose of a tiling strategy is to standardise the retrieval of a
window within a `.tif` image.

The interface for such classes is composed by two methods:
- `get_number_of_spatial_sample_per_image`: it returns the number of tiles that
        are extracted from the single `.tif` image, with respect to its size.
        It is used to calculate the length of a datasets that uses the specific
        tiling strategy.
- `get_window`: it returns the actual window as a tuple in the form of
        `(column, row, width, height)`

## Datasets
Datasets are defined in the `src/data/dataset.py` script and include
`SingleFolderImageDataset` (a pipeline for loading a folder of `.tif` images),
`LocationDataset` (to load whole events) and `NConsecutiveDataset` (to create
time series).

### SingleFolderImageDataset
The simplest form of dataset is given by the `SingleFolderImageDataset`, within
the `src/data/dataset.py` script.
The functionality of the dataset is to load the `.tif` images within a
specific folder, in particular `root_folder/folder_path` (see arguments).

The images can be filtered, normalised (see below for both) and queried through
a window from a tiling strategy class with the `get_datum` method.

Filters and normalisers are instatiated dynamically trhough the name of their
class and the args they needed.

#### Image filtering
Filtering is implemented in the `src/data/filters.py` script.
The aim of the classes defined there is to filter out image files depending on
specific criteria, like time position or respecting regular expressions.

#### Image normalisation
Images can be normalised through the normalisers classes defined in the
`src/data/normalisers.py` script.
In here, base normalisers can be composed through `ListNormaliser`s to create
more complex pipelines and through `CompositeNormaliser`s to apply different
operations to different channels of the image.

### LocationDataset
The `LocationDataset` dataset is used to create composite datasets from an event
folder, `main_folder`, which is then use within the single datasets, defined in
`dataset_specs`, to find the location the data are stored.

For each element of the `dataset_specs` list, a dataset is created and, at
sample time, one element per each of these dataset is retrieved and the same
window from the tiling strategy is applied to all of them.
The result is a list of elements, one for each dataset.

### NConsecutiveDataset
`NConsecutiveDataset` is used to extract time series from `LocationDataset`s.
The dataset for each index returns a list of sequential `LocationDataset`
samples, which in turn are a list of samples, one for each defined dataset.

## Datamodule
The main datamodule is `ParsedDataModule`, within the `src/data/datamodule.py`
script.
The datamodule is defined as a `LightningDataModule` class, and it creates its
train, validation and test datasets from a configuration dictionary.
Once the datamodule has been created, it is possible to save it through the
`pickle` library and load it if needed.

### Save and Load
Once configured (see `configuration.md`), it is possible to save a datamodule.
To do so, you can use the `src/data/make_datamodule.py` script, which also
lets you inspect the returned data.

The datamodule will be saved in a `cache_dir` with the `datamodule_name` given
in the configurtation file, which can then be used for loading it, as follows:

```
from src.data.datamodule import ParsedDataModule

data_module = ParsedDataModule.load_or_create(cfg_dict, cache_dir)
```
Once loaded, it is possible to modify the batch sizes and number of workers used
thorugh the method `data_module.set_batch_sizes_and_n_workers`.

This pipeline is very useful not only for evaluation scripts but especially for
jupyter notebooks, since the integration with the `hydra` configuration system
is not fully supported.
