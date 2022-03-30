import matplotlib
from pandas.core import indexing
import fsspec
from matplotlib import pyplot as plt 
import rasterio
from rasterio.plot import show 
import numpy as np 
import wandb
from glob import glob 
from timeit import default_timer as timer

def adjust_for_visualization(image_rgb, cut_off_value = 2000):
    image_rgb[image_rgb>cut_off_value] = cut_off_value
    image_rgb = (rasterio.plot.adjust_band(image_rgb, kind='linear')*255).astype(np.uint8)
    return image_rgb

def array_processing_vis(t, clip_max=2000):
    t = np.clip(t, np.nanmin(t), clip_max)
    t = (t - np.nanmin(t)) / np.nanmax(t) # between 0-1
    t = np.nan_to_num(t) # convert all nans to 0 only after we have scaled it ...
    z = (t * 255).astype(np.uint8) # 0-255

    return z

def s2_to_rgb(model_input_rgb_npy):
    # For inspiration we could see: https://github.com/spaceml-org/ml4floods/blob/f1cc17ef00a9748ed5ccc2e5206711d0ac023ffb/ml4floods/models/utils/uncertainty.py#L180
    model_input_rgb_npy = np.clip(model_input_rgb_npy / 3000., 0., 1.)
    return model_input_rgb_npy
