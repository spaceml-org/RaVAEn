"""
Run with
python3 -m scripts.gui_fewshot +training=simple_ae +dataset=samples_for_gui +module=simple_ae +project=gui +normalisation=log_scale +channels=rgb +checkpoint=/home/vitek/Vitek/python_codes/FDL21/gui_code/change-detection/epoch_09-step_1481.ckpt
 

"""

# Show the capabilities of clustering in feature space for few-shot learning tasks.
# ... best demonstrated with an interactive GUI ...

import cv2 
import numpy as np

from src.visualization.gui_utils import array_processing_vis, adjust_for_visualization, s2_to_rgb


###########################################
from tqdm import tqdm
import hydra
import torch
import wandb
import os 

from src.data.datamodule import ParsedDataModule
from src.utils import load_obj, deepconvert
from src.evaluation.evaluator import MetricEvaluator
from src.evaluation.utils import wandb_init, save_as_georeferenced_tif, visualize_unnormalised, as_plot
from src.evaluation.anomaly_functions import vae_anomaly_only_latents
from src.models.coversion_utils import save_model_json
from scipy.spatial import distance

import numpy as np
from pathlib import Path

def process_batch(inputs, nans_to_zeros=True):
    """Batch """
    # These are torch Tensors
    nan_masks = torch.isnan(inputs)
    bad_nuns = torch.any(nan_masks.view(nan_masks.shape[0], -1), dim=1)
    if bad_nuns.sum() > 0:
        print(f"Warning! We have {bad_nuns.sum()}/{len(bad_nuns)} samples with NaN values in the batch of inputs of the evaluation set.")

    if nans_to_zeros:
        inputs = torch.nan_to_num(inputs)  # Convert all NaN to 0s !

    return inputs

def project_windows_to_image_overlap(windows_array, grid_shape, overlap=24):
    # windows_array shape of ~ N, (channels), w, h
    tile_size = list(windows_array.shape[-2:])  # value of w,h
    tile_size[0] = tile_size[0] - overlap
    tile_size[1] = tile_size[1] - overlap
    channels = 1 if len(windows_array.shape) == 3 else windows_array.shape[1]
    image = np.zeros((channels, grid_shape[1]*tile_size[0], grid_shape[0]*tile_size[1]), dtype=np.float32)
    index2positions = {}
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
            index2positions[index] = [i*w,(i+1)*w, j*w,(j+1)*w]
            index += 1
            
    return image, index2positions

def load_data_from_dataloader(dataloader,module,overlap):
    if True:
        tiling_strat = dataloader.dataset.tiling_strategy # NConsecutiveDataset has a shared tiling strategy
        grid_shape = tiling_strat.grid_shape # for example: (8,6)
        
        all_unnormalized = []
        all_latents = []
        processed_inputs = []

        # Looping through windows in order:
        index = 0
        for batch in tqdm(dataloader):
            #   batch shape ~ 1 sequence, 2 inputs and unnormalized images, batch 32, ch 13, w h
            inputs = process_batch(batch[0][0]) # before
            inputs_unnormalized = batch[0][1]

            # These are numpy arrays
            # Anomaly map is pixel wise (as image), anomaly score is window based (as a single number)
            latents = vae_anomaly_only_latents(module, inputs, latents_keep_tensors=False)

            all_unnormalized += [inputs_unnormalized]
            all_latents += [latents.copy()]
            processed_inputs.append(inputs.cpu().detach().numpy())
            
            index += len(latents)
            ###print("index ~ ", len(all_latents)) # from 0 to 16, every time we load 32 ... 32*16 = 512 ...
            ###(hacky stop in the middle...)
            #print(index)
            if index > grid_shape[0] * grid_shape[1]: break

        processed_inputs = np.asarray(processed_inputs)
        processed_inputs = np.concatenate(processed_inputs)

        # Need to take care with input image to deal with variable input channels
        channels = all_unnormalized[0].shape[1]  # N, channels, w,h
        assert channels in [3, 13]
        if channels == 13:
            # We have all 13 channels
            all_unnormalized = np.concatenate(all_unnormalized)[:, [3, 2, 1]]  # select rgb -> remember zero index
        elif channels == 3:
            # Or just R,G,B in order
            all_unnormalized = np.concatenate(all_unnormalized)  # assumes uses ['B4','B3','B2'] in this order
        all_latents = np.concatenate(all_latents)


        # Conversions for plotting:
        final_image_unnormalized, _ = project_windows_to_image_overlap(all_unnormalized, grid_shape, overlap)
        final_image_unnormalized.transpose((1, 2, 0)) # (832, 1024, 3)
        final_image_processed, index2positions = project_windows_to_image_overlap(processed_inputs, grid_shape, overlap)
        final_image_processed.transpose((1, 2, 0))

        ## did for batch in tqdm(dataloader) go over it twice?
        all_latents = all_latents[0:grid_shape[0] * grid_shape[1]] 

        # Visualize as plots for Wandb:
        #final_image_unnormalized = visualize_unnormalised(final_image_unnormalized)
        #final_image_processed = as_plot(final_image_processed, "tmp2")
        #final_image_processed.show()

    return final_image_unnormalized, final_image_processed, all_latents, index2positions


@hydra.main(config_path='../config', config_name='config.yaml') # pip install hydra-core==1.1.0 omegaconf==2.1.0
def main(cfg):
    # Load configs
    cfg = deepconvert(cfg)

    # Load and setup the dataloaders
    data_module = \
        ParsedDataModule.load_or_create(cfg['dataset'], cfg['cache_dir'])

    if True:
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

    print("Current folder:", os.getcwd())

    model_identifier = str(module.model.__class__.__module__).replace('src.models.','')
    
    
    # Multiple datasets (each one is LocationDataset):
    val_loader_list = data_module.val_dataloader()
    if isinstance(val_loader_list, list):
        pass
    else:
        val_loader_list = [val_loader_list]
    test_loader_list = data_module.test_dataloader()
    if isinstance(test_loader_list, list):
        pass
    else:
        test_loader_list = [test_loader_list]

    for dataloader_index, dataloader in enumerate(test_loader_list):
        # Test serves as the eval image
        folder_name = dataloader.dataset.dataset.main_folder.stem
        print(f'Test Dataloader {dataloader_index} in {folder_name}')
        overlap = cfg['dataset']['test']['dataset_cls_args']['dataset_cls_args']['tiling_strategy_args']['overlap'][0]
        eval_raw, eval_processed, eval_latents, eval_index2positions = load_data_from_dataloader(dataloader, module, overlap)
    
    for dataloader_index, dataloader in enumerate(val_loader_list):
        # Val serves as the support image
        folder_name = dataloader.dataset.dataset.main_folder.stem
        print(f'Val Dataloader {dataloader_index} in {folder_name}')
        overlap = cfg['dataset']['valid']['dataset_cls_args']['dataset_cls_args']['tiling_strategy_args']['overlap'][0]
        support_raw, support_processed, support_latents, support_index2positions = load_data_from_dataloader(dataloader, module, overlap)

    # HAX on the same image?
    #eval_raw, eval_processed, eval_latents, eval_index2positions = support_raw, support_processed, support_latents, support_index2positions

    #######

    events = [i for i in dir(cv2) if 'EVENT' in i]
    print(events)

    class PainterHandler:
        def __init__(self, eval_raw, eval_processed, eval_latents, eval_index2positions, img_use, k_closest=4):
            self.eval_raw = eval_raw
            self.eval_processed = eval_processed
            self.img_use = img_use
            self.h, self.w, self.ch = self.eval_processed.shape
            self.eval_latents = eval_latents
            self.eval_index2positions = eval_index2positions

            self.k_closest = k_closest

            self.image_cv2 = None
            self.reset()

        def reset(self):
            eval_img = cv2.UMat(self.img_use) # Convert into CV2 format
            self.image_cv2 = eval_img

        def highlight_from_mean_latent(self, mean_latent):
            self.reset()
            mean_latents = np.repeat(np.expand_dims(mean_latent,0), len(self.eval_latents), 0)
            print("mean_latents.shape", mean_latents.shape, "self.eval_latents.shape", self.eval_latents.shape)
            
            distances = []
            indices = []
            for idx in range(len(self.eval_latents)):
                dst = distance.cosine(mean_latents[idx], self.eval_latents[idx])
                distances.append(dst)
                indices.append(idx)
            distances = np.asarray(distances)

            print("distances.shape = ", distances.shape)
            print("min, max = ", np.min(distances), np.max(distances))

            # Now highlight the nearest k=10 tiles ...
            distances_sorted, indices_sorted = zip(*sorted(zip(distances, indices)))
            for i in range(self.k_closest):
                print("selected idx", indices_sorted[i], "with distance", distances_sorted[i])
                position = self.eval_index2positions[indices_sorted[i]]

                thickness_multiplier = int(self.k_closest - i)
                self.highlight_one(position, thickness_multiplier, i)


        def highlight_one(self, position, thickness_multiplier, order):
            # position has numpy coordinates ~ image[:, i*w:(i+1)*w, j*w:(j+1)*w]  ~  [32, 64, 256, 288]
            # while x,y in cv2 are swapped
            y0, y1, x0, x1 = position

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image_cv2, str(order), (x0,y0), font, 0.6, (255,255,0), 1)
            cv2.rectangle(self.image_cv2, (x0,y0), (x1,y1), color=(0,255,255), thickness=thickness_multiplier)


    class ClickerHandler:
        def __init__(self, support_raw, support_processed, support_latents, support_index2positions, model, painter, support_img):
            self.tiles = []
            self.latents = []
            self.centers = []
            self.clicked_i = 0
            self.tile_size = 32
            
            self.support_raw = support_raw
            self.support_processed = support_processed
            self.support_latents = support_latents
            self.support_index2positions = support_index2positions

            self.model = model
            self.img = support_img

            self.painter = painter

            self.mode = 'exact_click'
            self.mode = 'nearest_tile'

        def reset(self, support_img):
            self.tiles = []
            self.latents = []
            self.centers = []
            self.clicked_i = 0
            self.img = support_img

        def click_event(self,event,x,y,flags,param):
            img = self.img
            font = cv2.FONT_HERSHEY_SIMPLEX

            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click -> draw a square!
                #print(x,",",y)
                self.centers.append([x,y])
                
                strXY = str(x)+", "+str(y)
                fontScale = 0.6
                cv2.putText(img, strXY, (x,y), font, fontScale, (255,255,0), 1)

                if self.mode == 'exact_click':
                    x0 = x - int(self.tile_size/2)
                    x1 = x + int(self.tile_size/2)
                    y0 = y - int(self.tile_size/2)
                    y1 = y + int(self.tile_size/2)
                    if x0 < 0 or x1 < 0 or y0 < 0 or y1 < 0:
                        return None # out of bounds ...

                    cv2.rectangle(img, (x0,y0), (x1,y1), color=(0,255,255), thickness=1)

                    # Version 1, sample tile from the clicked area:
                    tile = self.get_tile(x0,y0,x1,y1)
                    latent = self.tile2latent(tile)
                    #self.tiles.append(tile) # add into list of tiles
                    self.latents.append(latent)
                elif self.mode == 'nearest_tile':
                    # Version 2, load the latent from computations:
                    corresponding_index = self.get_position_index(x,y)
                    latent = self.support_latents[corresponding_index]
                    #print("latent ~ ", latent)
                    self.latents.append(latent)

                    y0, y1, x0, x1 = self.support_index2positions[corresponding_index]
                    #print("targetted ~ ", y0, y1, x0, x1)
                    cv2.rectangle(img, (x0,y0), (x1,y1), color=(0,255,255), thickness=1)


                # indexing directly into the picture:
                #print("clicked at:", y,x)

                self.final_update()
                # Next one please ...
                self.clicked_i += 1

        def show_debug(self, tile):
            cv2.namedWindow('debug')
            img = cv2.UMat(tile)
            cv2.imshow("debug", img)

        def get_tile(self, x0,y0,x1,y1):
            tile = self.support_processed[y0:y1,x0:x1,:]
            #self.show_debug(tile)
            #print("From", x0,y0,x1,y1, "tile.shape", tile.shape)
            return tile

        def tile2latent(self,tile):
            tile = torch.tensor(np.moveaxis(tile, -1, 0)).unsqueeze(0)
            latent = vae_anomaly_only_latents(self.model, tile, latents_keep_tensors=False)[0]
            return latent

        def get_position_index(self, x,y):
            # position has numpy coordinates ~ image[:, i*w:(i+1)*w, j*w:(j+1)*w]  ~  [32, 64, 256, 288]
            #                                  image[:, x0:x1, y0:y1]
            # while x,y in cv2 are swapped
            x, y = y, x

            def point_in_1d(x,y,a):
                if a >= x and a <= y:
                    return True
                return False

            self.support_index2positions
            for k in self.support_index2positions.keys():
                #position ~~ self.support_index2positions[k]
                x0,x1,y0,y1 = self.support_index2positions[k]

                overlaps = point_in_1d(x0,x1,x) and point_in_1d(y0,y1,y)
                if overlaps:
                    print("key", k, "overlaps!")
                    break

            return k

        def final_update(self):
            # Final update after a click
            # Embed all tiles!
            latents = self.latents.copy()
            """
            for tile_i, tile in enumerate(self.tiles):
                # Tile (np) to latent
                #print("tile as np", tile_i, tile.shape)
                tile = torch.tensor(np.moveaxis(tile, -1, 0)).unsqueeze(0)
                #print("tile as torch", tile_i, tile.shape)
                latent = vae_anomaly_only_latents(self.model, tile, latents_keep_tensors=False)[0]
                #print(latent.shape, latent)

                latents.append(latent)
            """

            # Get the mean vector:
            latents = np.asarray(latents)
            #print("all latents ~", latents.shape)

            mean_latent = np.mean(latents, 0)
            print("Mean latent ~", mean_latent, "from", latents.shape)
            self.painter.highlight_from_mean_latent(mean_latent)

    
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL) # < Windows scaling ON
    cv2.namedWindow('support_img')
    cv2.namedWindow('eval_img')

    ############################
    support_processed = np.moveaxis(support_processed, 0, -1)
    support_raw = np.moveaxis(support_raw, 0, -1)
    eval_processed = np.moveaxis(eval_processed, 0, -1)
    eval_raw = np.moveaxis(eval_raw, 0, -1)

    # Log normalised
    support_use = support_processed
    eval_use = eval_processed

    # Or raw
    support_use = s2_to_rgb(support_raw)
    eval_use = s2_to_rgb(eval_raw)
    ###############################

    support_img = cv2.UMat(support_use) # Convert into CV2 format
    eval_img = cv2.UMat(eval_use) # Convert into CV2 format
    
    painter = PainterHandler(eval_raw, eval_processed, eval_latents, eval_index2positions, eval_use)
    clicker_storage = ClickerHandler(support_raw, support_processed, support_latents, support_index2positions, module, painter, support_img)

    cv2.setMouseCallback('support_img',clicker_storage.click_event)

    normalisation_flag = True
    while True:
        # display the image and wait for a keypress
        eval_img = painter.image_cv2
        cv2.imshow("support_img", support_img)
        cv2.imshow("eval_img", eval_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            print("Clearing selection:")

            # Clear images:
            support_img = cv2.UMat(support_use) # Convert into CV2 format
            clicker_storage.reset(support_img)
            painter.reset()

        if key == ord("p"):
            print("Running debug:")
            import pdb; pdb.set_trace()            
            
        if key == ord("-") or key == ord("_"):
            # Less hits
            painter.k_closest -= 1
            if painter.k_closest == 0: painter.k_closest = 1

        if key == ord("+") or key == ord("="):
            # More hits
            painter.k_closest += 1
            
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    print("Selected Coordinates: ")
    for i in clicker_storage.centers:
        print(i)




if __name__ == '__main__':
    main()


