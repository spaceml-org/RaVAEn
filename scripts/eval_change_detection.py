"""
Typical command:
AE
python3 -m scripts.eval_change_detection +training=simple_ae +dataset=preliminary_sequential_bigger_multiEval_Germany +module=simple_ae +project=vae_single_location +normalisation=log_scale +checkpoint=/home/vit.ruzicka/results/vae_fullgrid/ae3nhema/checkpoints/epoch_09-step_1481.ckpt +channels=rgb

VAE
(to try ...)
python3 -m scripts.eval_change_detection +training=simple_vae +dataset=preliminary_sequential_bigger_multiEval_Germany +module=simple_vae +project=vae_single_location +normalisation=log_scale +checkpoint=/home/vit.ruzicka/branches/eval_branch/d8oflm4w/checkpoints/epoch_2949-step_2949.ckpt +channels=rgb


"""

from tqdm import tqdm
import hydra
import torch
import wandb
import os 

from src.data.datamodule import ParsedDataModule
from src.utils import load_obj, deepconvert
from src.evaluation.evaluator import MetricEvaluator
from src.evaluation.utils import wandb_init, save_as_georeferenced_tif, visualize_unnormalised, as_plot
from src.evaluation.anomaly_functions import vae_anomaly_function, grx_anomaly_function, vae_anomaly_function_with_latents
from src.models.coversion_utils import save_model_json
from scipy.spatial import distance

import numpy as np
from pathlib import Path


def process_batch(batch, nans_to_zeros=True):
    """Batch """
    # These are torch Tensors
    inputs = batch[0]
    change_maps = batch[1]  # shape: batch_num, channels, w, h ... contains: 0 ~ no change, 1 ~ change, 2 ~ invalid

    nan_masks = torch.isnan(inputs)
    bad_nuns = torch.any(nan_masks.view(nan_masks.shape[0], -1), dim=1)
    if bad_nuns.sum() > 0:
        print(f"Warning! We have {bad_nuns.sum()}/{len(bad_nuns)} samples with NaN values in the batch of inputs of the evaluation set.")

    if nans_to_zeros:
        inputs = torch.nan_to_num(inputs)  # Convert all NaN to 0s !

    invalid_mask = (change_maps == 2)[:, 0]  # shape: batch_num, w,h
    change_maps = change_maps.detach().cpu().numpy()

    return inputs, invalid_mask, change_maps


def project_windows_to_image_overlap(windows_array, grid_shape, overlap=24):
    # windows_array shape of ~ N, (channels), w, h
    tile_size = list(windows_array.shape[-2:])  # value of w,h
    tile_size[0] = tile_size[0] - overlap
    tile_size[1] = tile_size[1] - overlap
    channels = 1 if len(windows_array.shape) == 3 else windows_array.shape[1]
    image = np.zeros((channels, grid_shape[1]*tile_size[0], grid_shape[0]*tile_size[1]), dtype=np.float32)
    index = 0
    w = tile_size[0]  # change if tiles become not the same size
    for i in range(grid_shape[1]):
        for j in range(grid_shape[0]):
            if overlap == 0:
                topleft = windows_array[index]
            else:
                if len(windows_array[index].shape) == 3:  # N channels
                    topleft = windows_array[index, :, :-overlap, :-overlap]
                if len(windows_array[index].shape) == 2:  # 1 channel only
                    topleft = windows_array[index, :-overlap, :-overlap]

            image[:, i*w:(i+1)*w, j*w:(j+1)*w] = topleft
            index += 1
    return image


def project_windows_to_image_noOverlap(windows_array, grid_shape):
    # windows_array shape of ~ N, (channels), w, h
    tile_size = windows_array.shape[-2:]  # value of w,h
    channels = 1 if len(windows_array.shape) == 3 else windows_array.shape[1]
    image = np.zeros((channels, grid_shape[1]*tile_size[0], grid_shape[0]*tile_size[1]), dtype=np.float32)
    index = 0
    w = tile_size[0]  # change if tiles become not the same size
    for i in range(grid_shape[1]):
        for j in range(grid_shape[0]):
            image[:, i*w:(i+1)*w, j*w:(j+1)*w] = windows_array[index]
            index += 1
    return image


@hydra.main(config_path='../config', config_name='config.yaml')
def main(cfg):
    # Load configs
    cfg = deepconvert(cfg)

    # Load and setup the dataloaders
    data_module = \
        ParsedDataModule.load_or_create(cfg['dataset'], cfg['cache_dir'])

    if 'grx' not in cfg['module']['class']:  # VAE or AE

        cfg['module']['len_train_ds'] = data_module.len_train_ds
        cfg['module']['len_val_ds'] = data_module.len_val_ds
        cfg['module']['len_test_ds'] = data_module.len_test_ds
        cfg['module']['input_shape'] = (3, 32, 32)  # data_module.sample_shape_train_ds[0]

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
                                             cfg=cfg_module,
                                             train_cfg=cfg_train)
        module = module.cuda().eval()

    elif 'grx' in cfg['module']['class']:
        module = load_obj(cfg['module']['class'])()

        # Need to loop through all the data to train GRX before scoring in future loop
        # Need to loop through twice to fit the mean and then the correlation matrix
        for fitting in [module.partial_fit_mean, module.partial_fit_cov]:
            for batch in tqdm(data_module.test_dataloader()):
                inputs, _, _ = process_batch(batch, nans_to_zeros=False)
                # reshaping: batch, channels, w, h -> channels, N
                inputs = inputs.detach().cpu().numpy()
                inputs = inputs.transpose((1, 0, 2, 3))
                # NaNs need to be removed for GRX
                inputs = inputs.reshape(inputs.shape[0], -1)
                inputs = inputs[:, ~np.any(np.isnan(inputs), axis=0)]
                fitting(inputs)

        print("'Trained' GRX with:")
        module.report()

        anomaly_function = grx_anomaly_function
   
    evaluator = MetricEvaluator()  # computes metrics
    
    # Save model weights in non-version specific mode:
    print("Saving model weights into", checkpoint_root_path)

    model_identifier = str(module.model.__class__.__module__).replace('src.models.','')
    #torch.save(module.state_dict(), "model_"+model_identifier+".pt")
    save_model_json(module.model, checkpoint_root_path / f"model_{model_identifier}_latent_{cfg_module['model_cls_args']['latent_dim']}.json")
    exit(0)
    # Initializations for weights and biases:
    wandb_init(project_name="visualize_evaluation_v5_twin_vaes")
    # Visualizations table:
    table = wandb.Table(columns=["id", "input before", "input after", "prediction", "ground truth", "baseline"])
    columns = ["id", "area under precision curve", "precision at 100 recall", "efficiency over manual vetting", "spearman cor", "selected top 10% score"]
    score_table = wandb.Table(columns=columns)
    columns = ["id", "precision recall plot", "anomaly scatter plot"]
    plots_table = wandb.Table(columns=columns)

    # Multiple datasets (each one is LocationDataset):
    test_loader_list = data_module.test_dataloader()
    if isinstance(test_loader_list, list):
        pass
    else:
        test_loader_list = [test_loader_list]

    for dataloader_index, dataloader in enumerate(test_loader_list):
        predicted_anomaly_maps = []
        predicted_anomaly_scores = []
        true_anomaly_maps = []
        true_anomaly_fractions = []
        invalid_masks = []
        all_inputs_before = []
        all_inputs_after = []

        # baseline:
        baseline_anomaly_maps_all = []
        baseline_anomaly_scores_all = []

        folder_name = dataloader.dataset.dataset.main_folder.stem
        print(f'Evaluating on dataloader {dataloader_index} in {folder_name}')

        tiling_strat = dataloader.dataset.tiling_strategy # NConsecutiveDataset has a shared tiling strategy
        grid_shape = tiling_strat.grid_shape # for example: (8,6)

        # temporary saving of inputs ...
        tmp_save_inputs = False
        if tmp_save_inputs:
            processed_inputs_1 = []
            processed_inputs_2 = []

        # Looping through windows in order:
        for batch in tqdm(dataloader):
            # labeled:
            #   batch shape ~ 2 sequence, 3 inputs and change maps and unnormalized images, batch 32, ch 13, w h
            # unlabeled:
            #   batch shape ~ 2 sequence, 2 inputs and unnormalized images, batch 32, ch 13, w h
            fake_labels = len(batch[0]) == 2 # we have only "inputs and unnormalized images"

            inputs_before, invalid_mask_before, _ = process_batch(batch[-2]) # before
            inputs_after, invalid_mask_after, change_maps = process_batch(batch[-1]) # after

            if tmp_save_inputs:
                processed_inputs_1.append(inputs_before.cpu().detach().numpy())
                processed_inputs_2.append(inputs_after.cpu().detach().numpy())

            if fake_labels:
                # This dataset has no labels, fake the change_maps array...
                # We can't trust the scores, but can view the visualizations of predictions

                shape_of_images = list(batch[0][0].shape)
                shape_of_images[1] = 1 # change maps have only 1 channel
                change_maps = np.zeros(shape_of_images, dtype=np.float32)

                before_unnormalized = batch[-2][1] # second index points to normalized inputs, unnormalized inputs
                after_unnormalized = batch[-1][1]

            else:
                before_unnormalized = batch[-2][2] # second index points to normalized inputs, gt, unnormalized inputs
                after_unnormalized = batch[-1][2]

            # These are numpy arrays
            # Anomaly map is pixel wise (as image), anomaly score is window based (as a single number)
            anomaly_maps_before, anomaly_scores_before, latents_before = vae_anomaly_function_with_latents(module, inputs_before, invalid_mask_before, latents_keep_tensors=True)
            anomaly_maps_after, anomaly_scores_after, latents_after = vae_anomaly_function_with_latents(module, inputs_after, invalid_mask_after, latents_keep_tensors=True)

            # Distance between latents:
            anomaly_maps = np.ones_like(anomaly_maps_after)
            anomaly_scores = []
            for idx in range(len(inputs_before)):
                # difference_vector = latents_after[idx] - latents_before[idx]
                # dst = distance.euclidean(latents_after[idx], latents_before[idx])

                if "SimpleAE" in str(module.model.__class__):
                    # simple cosine distance
                    #dst = distance.cosine(latents_after[idx], latents_before[idx])
                    dst = distance.cosine(latents_after[idx].cpu().detach(), latents_before[idx].cpu().detach())


                elif "SimpleVAE" in str(module.model.__class__):
                    mu_1 = latents_before[0][idx]
                    log_var_1 = latents_before[1][idx]
                    mu_2 = latents_after[0][idx]
                    log_var_2 = latents_after[1][idx]

                    # Just distance between the means (wouldn't make use of the sigmas)
                    # v1 numpy
                    #dst = distance.cosine(mu_1, mu_2)
                    
                    # v1b tensor > numpy
                    #dst = distance.cosine(mu_1.cpu().detach(), mu_2.cpu().detach())

                    # v2 tensor
                    #cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    #dst = 1-cos(mu_1,mu_2)
                    #dst = float(dst.cpu().detach().numpy())

                    # v3 better metrics:
                    from src.evaluation.vae_metrics import KL_divergence_std, KL_divergence, wasserstein2
                    dst = KL_divergence(mu_1, log_var_1, mu_2, log_var_2)
                    #dst = KL_divergence_std(mu_1, log_var_1, mu_2, log_var_2)
                    #dst = wasserstein2(mu_1, log_var_1, mu_2, log_var_2)
                    dst = float(dst.cpu().detach().numpy())

                else:
                    assert False, "To be implemented..."

                anomaly_maps[idx] = anomaly_maps[idx] * dst
                anomaly_scores.append(dst)
            anomaly_scores = np.asarray(anomaly_scores)
            predicted_anomaly_maps += [anomaly_maps]
            predicted_anomaly_scores += [anomaly_scores]

            # Baseline in image space:
            baseline_anomaly_maps = np.ones_like(anomaly_maps_after)
            baseline_anomaly_scores = []
            for idx in range(len(inputs_before)):
                # Cosine distance in the raw image space:
                image_space_dst = distance.cosine(inputs_after[idx].flatten(), inputs_before[idx].flatten())

                baseline_anomaly_maps[idx] = baseline_anomaly_maps[idx] * image_space_dst
                baseline_anomaly_scores.append(image_space_dst)
            baseline_anomaly_scores = np.asarray(baseline_anomaly_scores)
            baseline_anomaly_maps_all += [baseline_anomaly_maps]
            baseline_anomaly_scores_all += [anomaly_scores]

            true_anomaly_maps += [change_maps[:, 0]]  # shape: batch_num, w,h
            invalid_mask = invalid_mask_after
            invalid_masks += [invalid_mask]
            all_inputs_before += [before_unnormalized]
            all_inputs_after += [after_unnormalized]

            true_anomaly_fraction = (change_maps==1).mean(axis=tuple(n for n in range(1, len(change_maps.shape))))
            true_anomaly_fractions += [true_anomaly_fraction]

        if tmp_save_inputs:
            processed_inputs_1 = np.asarray(processed_inputs_1)
            processed_inputs_2 = np.asarray(processed_inputs_2)
            processed_inputs_1 = np.concatenate(processed_inputs_1)
            processed_inputs_2 = np.concatenate(processed_inputs_2)
            np.save("processed_inputs_1.npy", processed_inputs_1)
            np.save("processed_inputs_2.npy", processed_inputs_2)

        # TODO: The following few lines concatenating take quite a long time in this script.
        # It would probably be better fill a numpy array than append to lists and then concat

        # Need to take care with input image to deal with variable input channels
        channels = all_inputs_before[0].shape[1]  # N, channels, w,h
        assert channels in [3, 13]
        if channels == 13:
            # We have all 13 channels
            all_inputs_rgb_before = np.concatenate(all_inputs_before)[:, [3, 2, 1]]  # select rgb -> remember zero index
            all_inputs_rgb_after = np.concatenate(all_inputs_after)[:, [3, 2, 1]]  # select rgb -> remember zero index
        elif channels == 3:
            # Or just R,G,B in order
            all_inputs_rgb_before = np.concatenate(all_inputs_before)  # assumes uses ['B4','B3','B2'] in this order
            all_inputs_rgb_after = np.concatenate(all_inputs_after)  # assumes uses ['B4','B3','B2'] in this order

        predicted_anomaly_maps = np.concatenate(predicted_anomaly_maps)
        true_anomaly_maps = np.concatenate(true_anomaly_maps)
        invalid_masks_maps = np.concatenate(invalid_masks)
        predicted_anomaly_scores = np.concatenate(predicted_anomaly_scores)
        true_anomaly_fractions = np.concatenate(true_anomaly_fractions)

        baseline_anomaly_maps_all = np.concatenate(baseline_anomaly_maps_all)
        baseline_anomaly_scores_all = np.concatenate(baseline_anomaly_scores_all)

        print(f'Predicted anomaly scores min {predicted_anomaly_scores.min()} max {predicted_anomaly_scores.max()}')

        area_under_precision_curve, precision_at_100_recall, efficiency_over_manual_vetting, pr_plot =\
            evaluator.evaluate_pixel_based(predicted_anomaly_maps, true_anomaly_maps, invalid_masks_maps)
        spearman_cor, selected_top_frac_score, anomaly_scatter_plot =\
            evaluator.evaluate_window_based(predicted_anomaly_scores, true_anomaly_fractions)

        # Conversions for plotting:
        overlap = cfg['dataset']['test']['dataset_cls_args']['dataset_cls_args']['tiling_strategy_args']['overlap'][0]
        final_image_rgb_before = project_windows_to_image_overlap(all_inputs_rgb_before, grid_shape, overlap).transpose((1, 2, 0))
        final_image_rgb_after = project_windows_to_image_overlap(all_inputs_rgb_after, grid_shape, overlap).transpose((1, 2, 0))
        final_prediction = project_windows_to_image_overlap(predicted_anomaly_maps, grid_shape, overlap).transpose((1, 2, 0))
        final_ground_truth = project_windows_to_image_overlap(true_anomaly_maps, grid_shape, overlap).transpose((1, 2, 0))

        final_prediction_baseline = project_windows_to_image_overlap(baseline_anomaly_maps_all, grid_shape, overlap).transpose((1, 2, 0))

        final_prediction[final_ground_truth == 2] = np.nan
        final_prediction_baseline[final_ground_truth == 2] = np.nan

        # Save as GeoTIFF!
        save_geotif_path = "prediction_geotif_"+folder_name+".tif"  # will be in the hydra folder
        # for example: /home/vit.ruzicka/branches/eval_branch/change-detection/outputs/<day>/<time>/
        sample_input_tif_path = str(dataloader.dataset.dataset.datasets[0].tifs[-1]) # last tif
        save_as_georeferenced_tif(final_prediction, save_geotif_path, sample_input_tif_path)

        # Visualize as plots for Wandb:
        final_image_rgb_before = visualize_unnormalised(final_image_rgb_before)
        final_image_rgb_after = visualize_unnormalised(final_image_rgb_after)
        final_prediction = as_plot(final_prediction, "tmp2")
        final_prediction_baseline = as_plot(final_prediction_baseline, "tmp2")
        final_ground_truth = as_plot(final_ground_truth, "tmp3")

        before = wandb.Image(final_image_rgb_before)
        after = wandb.Image(final_image_rgb_after)
        prediction = wandb.Image(final_prediction)
        baseline = wandb.Image(final_prediction_baseline)
        # Trying to mask the images (fails with Figures ...)
        # prediction = wandb.Image(final_image_rgb, masks={"prediction" : {"mask_data": final_prediction}})
        gt = wandb.Image(final_ground_truth)
        row = [0, before, after, prediction, gt, baseline]
        table.add_data(*row)

        # Scores table:
        row = [0, area_under_precision_curve, precision_at_100_recall, efficiency_over_manual_vetting, spearman_cor, selected_top_frac_score]
        score_table.add_data(*row)

        # Plots table:
        row = [0, wandb.Image(pr_plot), wandb.Image(anomaly_scatter_plot)]
        plots_table.add_data(*row)

    wandb.log({"map_table": table})
    wandb.log({"score_table": score_table})
    wandb.log({"plots_table": plots_table})

    wandb.finish()


if __name__ == '__main__':
    main()


