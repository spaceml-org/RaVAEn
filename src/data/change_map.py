from visualization_utils import file_exists, visualize_rgb
from filter_utils import filter_file_list
import fsspec
import rasterio
from rasterio.plot import show 
from matplotlib import pyplot as plt 

import numpy as np
import save_cog

def open_prediction(tif_path):
    src = rasterio.open("gs://" + tif_path)
    image = src.read()
    #print(image.shape)
    return image, src

def save_change_map(change_map, file_name, profile):
    # Saves contents of the change_map array into file_name, uses GEOTiff profile (ideally from the previous files ~ profile = src.profile)
    save_cog.save_cog(change_map, file_name, profile=profile)
    

def debug_save_as_plot(arr,name):
    plot = plt.figure()
    tmp = (rasterio.plot.adjust_band(arr, kind='linear')*255).astype(np.uint8)
    rasterio.plot.show(tmp)
    plot.savefig(name)
    plt.close()


def change_map_for_pair(fs, image_path_1, image_path_2, debug=False, prediction_folder = "/segmentation/"):
    # Generates a change map for a pair of images.
    # (Note: This function can later be moved somewhere more sensible)
    
    print("comparing", image_path_1, "<>", image_path_2)

    #prediction_folder = "/WFV1_unet/"
    prediction_path_1 = image_path_1.replace("/S2/", prediction_folder)
    prediction_path_2 = image_path_2.replace("/S2/", prediction_folder)

    if file_exists(fs, prediction_path_1) and file_exists(fs, prediction_path_2):
        
        prediction_1, src = open_prediction(prediction_path_1)
        prediction_2, _ = open_prediction(prediction_path_2)
        assert prediction_1.shape == prediction_2.shape

        if debug: debug_save_as_plot(prediction_1, "pred1.png")

        # Predictions contain: {0: invalid, 1:land, 2: water, 3:cloud}
        water_1 = np.zeros_like(prediction_1)
        water_1[prediction_1 == 2] = 1

        water_2 = np.zeros_like(prediction_2)
        water_2[prediction_2 == 2] = 1

        if debug: debug_save_as_plot(water_1, "water1.png")
        if debug: debug_save_as_plot(water_2, "water2.png")

        changes = np.zeros_like(prediction_1)
        got_water = np.where( np.logical_and( prediction_1 == 2, prediction_2 != 2) ) # from water to not water
        changes[got_water] = 1
        lost_water = np.where( np.logical_and( prediction_1 != 2, prediction_2 == 2) ) # from not water to water
        changes[lost_water] = 1
        if debug: debug_save_as_plot(changes, "changes.png")

        uncertain = np.zeros_like(prediction_1)
        select_indices = np.where( np.logical_or( prediction_1 == 0, prediction_1 == 3) )
        uncertain[select_indices] = 1 # anything in prediction 1 which includes clouds or invalid pixels
        select_indices = np.where( np.logical_or( prediction_2 == 0, prediction_2 == 3) )
        uncertain[select_indices] = 1 # anything in prediction 1 which includes clouds or invalid pixels
        if debug: debug_save_as_plot(uncertain, "uncertain.png")


        # Difference maps contains: {0: no change, 1: change , 2: ignore label}

        changes[uncertain == 1] = 2
        if debug: debug_save_as_plot(changes, "change_map.png")

        return changes, src

    else:
        print("Prediction file not available!")
        print(file_exists(fs, prediction_path_1), prediction_path_1)
        print(file_exists(fs, prediction_path_2), prediction_path_2)
        return None, None

        

def generate_change_maps(root_directory):
    fs = fsspec.filesystem("gs")
    dirs = fs.glob(root_directory+"/*")
    for dir in dirs:
        tif_files = fs.glob(dir+"/S2/*.tif")
        tif_files  = sorted(tif_files)[-2:] # last image is after the event, second to last the earliest before
        print(dir, "len tif_files", len(tif_files))

        if len(tif_files)!=2:
            print(f"Not enough tifs for directory {dir}")
            continue

        changes, src = change_map_for_pair(fs, tif_files[0], tif_files[1], debug=False)

        if changes is None or src is None:
            print(f"Not enough tifs for directory {dir}")
            continue
            
        file_name = "gs://" + tif_files[-1].replace("/S2/", "/changes/")
        
        # save the changes
        save_change_map(changes, file_name, src.profile)

if __name__=="__main__":
    fs = fsspec.filesystem("gs")
    
    """
    # Only on filtered, "well behaving" folders:
    root_folder = "gs://fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/*/*.csv"
    _, filtered_folders, list_of_files = filter_file_list(fs, root_folder, subset=50)
    print("Prefiltered", len(filtered_folders), "folders.")

    for demo_folder_csv in list_of_files:
        print(demo_folder_csv.filename)
        if len(demo_folder_csv.filename) > 1: # at least two
            image_path_1 = demo_folder_csv.filename.iloc[0]            
            image_path_2 = demo_folder_csv.filename.iloc[-1]

         change_map_for_pair(fs, image_path_1, image_path_2)

    image_path_1 = "fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/EMSR260_01CICOGNARA_DEL_MONIT01_v2_observed_event_a/2017-11-18.tif"
    image_path_2 = "fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/EMSR260_01CICOGNARA_DEL_MONIT01_v2_observed_event_a/2017-12-13.tif"
    generate_change_maps("fdl-ml-payload/worldfloods_change_TestDownload3/train/S2")
    """

    #root_directory = "gs://fdl-ml-payload/worldfloods_change_singleScene/train/S2"
    root_directory = "gs://fdl-ml-payload/worldfloods_change_TestDownload3/train/S2"
    root_directory = "gs://fdl-ml-payload/worldfloods_change_Germany_multiScene_v01"
    generate_change_maps(root_directory)