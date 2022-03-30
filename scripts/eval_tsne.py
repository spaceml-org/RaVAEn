"""
Typical command:
python3 -m scripts.eval_tsne +training=simple_ae +dataset=preliminary_sequential_bigger_multiEval_Germany +module=simple_ae +project=vae_single_location +normalisation=log_scale +checkpoint=/home/vit.ruzicka/results/vae_fullgrid/ae3nhema/checkpoints/last.ckpt +channels=rgb

"""

import re
from tqdm import tqdm
import hydra
import torch
import wandb

from src.data.datamodule import ParsedDataModule
from src.utils import load_obj, deepconvert
from src.evaluation.evaluator import MetricEvaluator
from src.evaluation.anomaly_functions import vae_anomaly_function, grx_anomaly_function, vae_anomaly_function_with_latents
from src.evaluation.utils import as_plot, prepare_image, wandb_init, visualize_unnormalised, save_as_georeferenced_tif
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from scripts.eval_change_detection import process_batch, project_windows_to_image_noOverlap, project_windows_to_image_overlap

import numpy as np
from pathlib import Path
import PIL
from PIL import Image, ImageDraw
from PIL import Image, ImageOps
import random

def function_tsne(features, tsne_lr, tsne_perpexity):
    tsne = TSNE(n_components=2, learning_rate=tsne_lr, perplexity=tsne_perpexity, angle=0.2, verbose=2).fit_transform(features)
    print("tsne.shape",tsne.shape) # tsne.shape (255, 2)
    return tsne

def function_umap(features):
    try:
        from umap import UMAP
    except:
        print("Will have to install UMAP with: !pip install umap-learn")
        assert False
    umap_vals = UMAP(n_components=2).fit_transform(features)
    print("umap_vals.shape",umap_vals.shape) # tsne.shape (255, 2)
    return umap_vals

def array_processing_vis(t, clip_max=2000):
        # For inspiration we could see: https://github.com/spaceml-org/ml4floods/blob/f1cc17ef00a9748ed5ccc2e5206711d0ac023ffb/ml4floods/models/utils/uncertainty.py#L180

        t = np.clip(t, np.nanmin(t), clip_max)
        t = (t - np.nanmin(t)) / np.nanmax(t) # between 0-1
        t = np.nan_to_num(t) # convert all nans to 0 only after we have scaled it ...
        z = (t * 255).astype(np.uint8) # 0-255

        return z
        #return Image.fromarray(z)

def visualize_embedding(latents, images, optional_gts=None, optional_preds=None, name_prefix="", PCA_first=True, subset=None,
                    method = "tsne",
                    tsne_perpexity = 30, tsne_lr = 150, threshold_gt = 0.5, threshold_pred = 0.5):
    # Generates a TSNE plot using the provided images
    # inputs:
    # - latents shape (255, 1024)
    # - images shape (255, 3, 32, 32)

    print("==== TSNE visualization ====")
    print("latents",latents.shape)
    print("images",images.shape)

    features = np.array(latents)

    if PCA_first:
        pca = PCA(n_components=100) # n_components must be lower than the number of samples ...
        pca.fit(features)
        pca_features = pca.transform(features)
        print("pca_features.shape",pca_features.shape)
        features = pca_features


    if subset is not None and len(images) > subset:
        # subset and shuffle
        sort_order = list(random.sample(range(len(images)), subset))
    else:
        # shuffle
        sort_order = list(random.sample(range(len(images)), len(images)))
    images = [images[i] for i in sort_order]
    features = [features[i] for i in sort_order]

    if optional_gts is not None:
        optional_gts = [optional_gts[i] for i in sort_order]
    if optional_preds is not None:
        optional_preds = [optional_preds[i] for i in sort_order]

    # TODO: as an argument instead ...
    if method == 'tsne':
        embedding = function_tsne(features, tsne_lr, tsne_perpexity)
    elif method == 'umap':
        embedding = function_umap(features)
    else:
        print("Method", method, "not supported!")
        assert False

    tx, ty = embedding[:,0], embedding[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    width = 600 # 4000
    height = 500 # 3000
    max_dim = 32 # tile size # 100
    thickness = 4 # to mark gt/pred tiles

    full_image = Image.new('RGBA', (width, height))
    idx = 0
    for img, x, y in tqdm(zip(images, tx, ty)):
        z = np.moveaxis(img, 0, -1)
        tile = Image.fromarray(z)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)

        draw_tile = ImageDraw.Draw(tile, "RGBA")
        # if we have predictions and gts:
        if optional_gts is not None:
            gt = optional_gts[idx]

            if gt > threshold_gt:
                # Mark this one with red:
                for i in range(0, thickness):
                    draw_tile.rectangle([(i, i),(max_dim-i,max_dim-i) ], fill=(255,255,255,0), outline="red")

        if optional_preds is not None:
            pred = optional_preds[idx]

            if pred > threshold_pred:
                # Mark this one with green:
                for i in range(thickness, int(2*thickness)):
                    draw_tile.rectangle([(i, i),(max_dim-i,max_dim-i) ], fill=(255,255,255,0), outline="green")

        idx += 1

        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

    #"/home/vit.ruzicka/branches/eval_branch/change-detection/tsne_plots/"
    full_image.save(name_prefix+"_"+method+"-all.png")

    # Also show as a 2D grid:
    tsne_as_2d_grid(embedding, images, optional_gts, optional_preds, name_prefix, threshold_gt, threshold_pred, method)

def tsne_as_2d_grid(embedding, images, optional_gts=None, optional_preds=None, name_prefix="", threshold_gt = 0.5, threshold_pred = 0.5, method='tsne'):
    # Needs: !pip install git+https://github.com/Quasimondo/RasterFairy
    try:
        import rasterfairy
    except:
        print("Install RasterFairy with: !pip install git+https://github.com/Quasimondo/RasterFairy")
        assert False
    import math

    # nx * ny = 256, the number of images
    total_n = len(images)
    nx = int(math.sqrt(total_n))
    ny = int(total_n / nx)
    can_display = int(nx * ny)

    embedding = embedding[:can_display]
    images = images[:can_display]

    # assign to grid
    grid_assignment = rasterfairy.transformPointCloud2D(embedding, target=(nx, ny))

    tile_width = 32
    tile_height = 32
    max_dim = max(tile_width, tile_height)
    thickness = 4 # to mark gt/pred tiles
    full_width = tile_width * nx
    full_height = tile_height * ny

    print("will make grid:", full_width, "x", full_height)

    aspect_ratio = float(tile_width) / tile_height

    grid_image = Image.new('RGB', (full_width, full_height))

    idx = 0
    for img, grid_pos in tqdm(zip(images, grid_assignment[0])):
        #print(grid_pos)
        idx_x, idx_y = grid_pos
        x, y = tile_width * idx_x, tile_height * idx_y
        z = np.moveaxis(img, 0, -1)
        tile = Image.fromarray(z)
        tile_ar = float(tile.width) / tile.height  # center-crop the tile to match aspect_ratio
        if (tile_ar > aspect_ratio):
            margin = 0.5 * (tile.width - aspect_ratio * tile.height)
            tile = tile.crop((margin, 0, margin + aspect_ratio * tile.height, tile.height))
        else:
            margin = 0.5 * (tile.height - float(tile.width) / aspect_ratio)
            tile = tile.crop((0, margin, tile.width, margin + float(tile.width) / aspect_ratio))


        draw_tile = ImageDraw.Draw(tile, "RGBA")
        # if we have predictions and gts:
        if optional_gts is not None:
            gt = optional_gts[idx]

            if gt > threshold_gt:
                # Mark this one with red:
                for i in range(0, thickness):
                    draw_tile.rectangle([(i, i),(max_dim-i,max_dim-i) ], fill=(255,255,255,0), outline="red")

        if optional_preds is not None:
            pred = optional_preds[idx]

            if pred > threshold_pred:
                # Mark this one with green:
                for i in range(thickness, int(2*thickness)):
                    draw_tile.rectangle([(i, i),(max_dim-i,max_dim-i) ], fill=(255,255,255,0), outline="green")
        idx += 1

        tile = tile.resize((tile_width, tile_height), Image.ANTIALIAS)
        grid_image.paste(tile, (int(x), int(y)))

    #"/home/vit.ruzicka/branches/eval_branch/change-detection/tsne_plots/"+
    grid_image.save(name_prefix+"_"+method+"-grid.jpg")


@hydra.main(config_path='../config', config_name='config.yaml')
def main(cfg):
    # Load configs
    cfg = deepconvert(cfg)

    # Load and setup the dataloaders
    data_module = ParsedDataModule.load_or_create(cfg['dataset'], cfg['cache_dir'])
    data_module.set_batch_sizes_and_n_workers(
        cfg['training']['batch_size_train'],
        cfg['training']['num_workers'],
        cfg['training']['batch_size_valid'],
        cfg['training']['num_workers'],
        cfg['training']['batch_size_test'],
        cfg['training']['num_workers']
    )

    if 'grx' not in cfg['module']['class']: # VAE or AE

        cfg['module']['len_train_ds'] = data_module.len_train_ds
        cfg['module']['len_val_ds'] = data_module.len_val_ds
        cfg['module']['len_test_ds'] = data_module.len_test_ds
        cfg['module']['input_shape'] = (3, 32, 32) #data_module.sample_shape_train_ds[0]

        cfg_train = cfg['training']
        cfg_module = cfg['module']

        module = load_obj(cfg['module']['class'])(cfg_module, cfg_train)

        checkpoint_root_path = Path(cfg['checkpoint']).parent.parent
        hparams_path = checkpoint_root_path / 'hparams.yaml'
        checkpoint_path = cfg['checkpoint']

        cfg['module']['init_weights'] = False

        print(f'Loading checkpoint {checkpoint_path}')
        module = module.load_from_checkpoint(str(checkpoint_path),
                                            hparams_file=str(hparams_path),
                                            cfg=cfg_module, train_cfg=cfg_train)
        module = module.cuda().eval()


    # Multiple datasets (each one is LocationDataset with single loaded folder):
    if 'keep_separate' in cfg['dataset']['test'].keys() and cfg['dataset']['test']['keep_separate']:  # keep_separate: true
        test_loader_list = data_module.test_dataloader()
    else:
        test_loader_list = [data_module.test_dataloader()]

    for dataloader_index, dataloader in enumerate(test_loader_list):
        all_inputs_before = []
        all_inputs_after = []
        all_latents_before = []
        all_latents_after = []

        predicted_anomaly_scores = []
        true_anomaly_fractions = []

        folder_name = str(dataloader.dataset.dataset.main_folder).split("/")[-1]
        print("Evaluating on dataloader #", dataloader_index, ":", dataloader, "in folder", folder_name)

        # Looping through windows in order:
        for batch in tqdm(dataloader):
            # labeled:
            #   batch shape ~ 2 sequence, 3 inputs and change maps and unnormalized images, batch 32, ch 13, w h
            # unlabeled:
            #   batch shape ~ 2 sequence, 2 inputs and unnormalized images, batch 32, ch 13, w h
            fake_labels = len(batch[0]) == 2 # we have only "inputs and unnormalized images"

            inputs_before, invalid_mask_before, _ = process_batch(batch[-2]) # before
            inputs_after, invalid_mask_after, change_maps = process_batch(batch[-1]) # after

            if fake_labels:
                before_unnormalized = batch[-2][1] # second index points to normalized inputs, unnormalized inputs
                after_unnormalized = batch[-1][1]
            else:
                before_unnormalized = batch[-2][2] # second index points to normalized inputs, gt, unnormalized inputs
                after_unnormalized = batch[-1][2]

            # These are numpy arrays
            # Anomaly map is pixel wise (as image), anomaly score is window based (as a single number)
            _, _, latents_before = vae_anomaly_function_with_latents(module, inputs_before, invalid_mask_before)
            _, _, latents_after = vae_anomaly_function_with_latents(module, inputs_after, invalid_mask_after)

            # For TSNE we care about:
            # - latents
            # - original images

            # Convert tile scores into map for visualization
            anomaly_scores = []
            for idx in range(len(latents_before)):
                #difference_vector = latents_after[idx] - latents_before[idx]
                #dst = distance.euclidean(latents_after[idx], latents_before[idx])
                dst = distance.cosine(latents_after[idx], latents_before[idx])
                anomaly_scores.append(dst)

            anomaly_scores = np.asarray(anomaly_scores)
            predicted_anomaly_scores += [anomaly_scores]

            true_anomaly_fraction = (change_maps==1).mean(axis=tuple(n for n in range(1, len(change_maps.shape))))
            true_anomaly_fractions += [true_anomaly_fraction]

            all_inputs_before += [before_unnormalized]
            all_inputs_after += [after_unnormalized]
            all_latents_before += [latents_before]
            all_latents_after += [latents_after]

        # TODO: The following few lines concatenating take quite a long time in this script.
        # It would probably be better fill a numpy array than append to lists and then concat

        # Need to take care with input image to deal with variable input channels
        channels = all_inputs_before[0].shape[1] # N, channels, w,h
        assert channels in [3, 13]
        if channels == 13:
            # We have all 13 channels
            all_inputs_rgb_before = np.concatenate(all_inputs_before)[:,[3,2,1]] # select rgb -> remember zero index
            all_inputs_rgb_after = np.concatenate(all_inputs_after)[:,[3,2,1]] # select rgb -> remember zero index
        elif channels==3:
            # Or just R,G,B in order
            all_inputs_rgb_before = np.concatenate(all_inputs_before)  #  assumes uses ['B4','B3','B2'] in this order
            all_inputs_rgb_after = np.concatenate(all_inputs_after)  #  assumes uses ['B4','B3','B2'] in this order

        all_latents_before = np.concatenate(all_latents_before)
        all_latents_after = np.concatenate(all_latents_after)

        predicted_anomaly_scores = np.concatenate(predicted_anomaly_scores)
        true_anomaly_fractions = np.concatenate(true_anomaly_fractions)

        print("Images:")
        print("all_inputs_rgb_before",all_inputs_rgb_before.shape)
        print("all_inputs_rgb_after",all_inputs_rgb_after.shape)
        print("Latents:")
        print("all_latents_before",all_latents_before.shape)
        print("all_latents_after",all_latents_after.shape)
        print("Scores:")
        print("predicted_anomaly_scores",predicted_anomaly_scores.shape) # We should threshold this to select only some of the samples ...
        print("true_anomaly_fractions",true_anomaly_fractions.shape)

        # Normalize in between 0-1 => then use threshold?
        true_anomaly_fractions = (true_anomaly_fractions - np.min(true_anomaly_fractions))/np.ptp(true_anomaly_fractions)
        predicted_anomaly_scores = (predicted_anomaly_scores - np.min(predicted_anomaly_scores))/np.ptp(predicted_anomaly_scores)
        threshold_gt=np.mean(true_anomaly_fractions)
        threshold_pred=np.mean(predicted_anomaly_scores)
        print("We have normalized the prediction scores and are using thresholds of threshold_gt=",threshold_gt, "threshold_pred=",threshold_pred)

        # To ignore predictions and labels, use:
        true_anomaly_fractions, predicted_anomaly_scores=None, None

        normalized_rgb = array_processing_vis(all_inputs_rgb_after.copy(), clip_max=2000)
        visualize_embedding(all_latents_after, normalized_rgb, optional_gts=true_anomaly_fractions, optional_preds=predicted_anomaly_scores,
                        threshold_gt=threshold_gt, threshold_pred=threshold_pred,
                        method = "tsne",
                        name_prefix=str(dataloader_index).zfill(2)+"_perp_"+str(30), tsne_perpexity=30) # 30 looks nice
        visualize_embedding(all_latents_after, normalized_rgb, optional_gts=true_anomaly_fractions, optional_preds=predicted_anomaly_scores,
                        threshold_gt=threshold_gt, threshold_pred=threshold_pred,
                        method = "umap",
                        name_prefix=str(dataloader_index).zfill(2))


if __name__ == '__main__':
    main()
