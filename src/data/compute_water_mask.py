from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.data.worldfloods import dataset
import torch
import rasterio
from src.data import save_cog
import numpy as np
import fsspec
from multiprocessing import Pool, Lock
from datetime import datetime


MODEL_NAME = "WFV1_unet"

def load_inference_function():

    experiment_name = "WFV1_unet"
    config_fp = f"gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart/{experiment_name}/config.json"
    config = get_default_config(config_fp)

    # The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory constrained environment set this value to 128
    config["model_params"]["max_tile_size"] = 1024

    config["model_params"]['model_folder'] = 'gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart'
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)
    model.to("cuda")
    channels = get_channel_configuration_bands(config.model_params.hyperparameters.channel_configuration)

    return get_model_inference_function(model, config,apply_normalization=True), channels



@torch.no_grad()
def get_segmentation_mask(torch_inputs, inference_function):
    outputs = inference_function(torch_inputs.unsqueeze(0))[0]
    prediction = torch.argmax(outputs, dim=0).long()
    mask_invalid = torch.all(torch_inputs == 0, dim=0)
    prediction += 1
    prediction[mask_invalid] = 0  # (H, W) {0: invalid, 1:land, 2: water, 3:cloud}
    prediction = prediction.unsqueeze(0)
    return np.array(prediction.cpu()).astype(np.uint8)

def parallel_fun(x):
    filename = f"gs://{x}"
    ret =  dataset.load_input(filename, window=None, channels=channels), filename
    return ret



if __name__ == "__main__":
    inference_function, channels = load_inference_function()
    fs = fsspec.filesystem("gs")
    tiff_files = fs.glob("gs://fdl-ml-payload/worldfloods_change_TestDownload3/*/S2/*/*.tif")
    tiff_files = [f for f in tiff_files if not fs.exists(f"gs://{f}".replace("/S2/", "/segmentation/"))]

    num_workers = 0
    with torch.no_grad():
        total = 0
        for (torch_inputs, transform), filename in map(parallel_fun, tiff_files):

            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total}/{len(tiff_files)}) Processing {filename}")
            total+=1

            filename_save = filename.replace("/S2/", "/segmentation/")

            if fs.exists(filename_save):
                continue

            with rasterio.open(filename) as src:
                crs = src.crs

            prediction = get_segmentation_mask(torch_inputs, inference_function)

            profile = {"crs": crs, "transform": transform, "compression": "lzw", "RESAMPLING":"NEAREST","nodata": 0}

            save_cog.save_cog(prediction, filename_save, profile=profile,
                              tags={"model": MODEL_NAME})



