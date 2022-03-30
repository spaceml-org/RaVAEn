import wandb
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show

import torch
from torchvision.transforms import ToPILImage

from src.data.save_cog import save_cog
from src.utils import load_obj


def as_plot(image, name, save=False):
    #assert image.shape[-1] in [1, 3]

    fig = plt.figure(dpi=600)
    plt.imshow(image)
    plt.colorbar()

    if save:
        plt.savefig(name+".png")
    return fig


def wandb_init(project_name="visualize_evaluation"):
    wandb.login()
    wandb.init(project=project_name, config={})


def prepare_image(x):
    # we have some very high numbers in the inputs ...
    return ((x - x.min()) * 255).astype(np.uint8)


def visualize_unnormalised(image_array,
                           cut_off_value=2000,
                           show=False,
                           save='tmp.png'):
    # Visualization in the original unnormalized format

    plot = plt.figure()

    image_array[image_array > cut_off_value] = cut_off_value
    image_array = (rasterio.plot.adjust_band(image_array, kind='linear') * 255)
    image_array = image_array.astype(np.uint8)
    array = np.moveaxis(image_array, -1, 0)

    rasterio.plot.show(array)
    plot.tight_layout()
    plt.axis('off')

    if show:
        plot.show()  # only on non-vm machines
    if save:
        plot.savefig(save)

    return plot


def save_as_georeferenced_tif(image_array, save_name, source_tif):
    # Loads the profile from the source tif and then saves the (probably
    # predictions) image array into georeferenced tif

    try:  # paths on bucket
        src = rasterio.open("gs://" + source_tif)
    except:  # paths on local
        src = rasterio.open(source_tif)

    # Juggle the shapes to maintain the same dimensionality ...
    array = np.moveaxis(image_array, -1, 0)  # for example ~ (1, 1984, 1376)

    # Fake the same dimensional array as is expected ...
    Z = np.zeros((src.profile["count"], src.profile["height"],
                  src.profile["width"]))  # for example ~ (15, 1995, 1403)
    Z[0][0:array.shape[1], 0:array.shape[2]] = array

    save_cog(Z, save_name, profile=src.profile)


def tassellate(tiles, nw, nh, overlap=[16, 16]):
    total_len, c, h, w = tiles.shape
    n_tiles_per_image = nw * nh

    b = int(total_len / n_tiles_per_image)
    assert b - total_len / n_tiles_per_image == 0, f"tile shape: {tiles.shape}, tiles per image: {n_tiles_per_image}"

    h1 = overlap[0] // 2
    h2 = h - overlap[0] // 2
    dh = h2 - h1
    assert dh > 0

    w1 = overlap[1] // 2
    w2 = w - overlap[1] // 2
    dw = w2 - w1
    assert dw > 0

    result = torch.zeros(b, c, dh * nh, dw * nw)

    for i in range(b):
        sliced_tile = \
            tiles[i * n_tiles_per_image: (i+1) * n_tiles_per_image, :, h1:h2, w1:w2]

        for j in range(nh):
            for k in range(nw):
                result[i, :, j * dh:(j+1) * dh, k*dw:(k+1) * dw] = \
                    sliced_tile[j * nw + k]
    return result


def get_eval_result(method, is_vae, **kwargs):
    if hasattr(method, 'for_vae'):
        s1 = '' if method.for_vae else 'not '
        s2 = '' if is_vae else 'not '
        assert method.for_vae == is_vae, f"Method is {s1}for VAEs but this is {s2}a VAE"

    result = method(is_vae=is_vae, **kwargs)
    return result
