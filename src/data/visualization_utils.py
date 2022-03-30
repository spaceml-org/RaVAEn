import matplotlib
from pandas.core import indexing
from filter_utils import load_csv_with_file_links, filter_file_list
import fsspec
from matplotlib import pyplot as plt 
import rasterio
from rasterio.plot import show 
import numpy as np 
import wandb
from glob import glob 
from timeit import default_timer as timer

# TODO: Maybe also show near infra-red also, good for water ~ will be black prolly
#       (12, 8, 4) ~ good enough info for water vs ground vs vegetation => maybe better visualization
#       or NDWI ~ https://foodsecurity-tep.net/S2_NDWI
# TODO: Add index

def folder_has_images(fs, folder, k = 1):
    files = fs.glob(folder+"/*")
    images = [f for f in files if f[-4:] == ".tif"]
    return len(images) > k

"""
import cv2
def visualize_rgb_fast(tif_path, cut_off_value = 2000, show=False, save="tmp.png"):
    # CV2 or SKIMAGE doesnt read from GS

    #from skimage import io
    
    #src = io.imread("gs://" + tif_path)

    src = rasterio.open("gs://" + tif_path) # Can I open fast in lower resolution?
    print(dir(src))
    
    print(src)
    return None, None #plot, resolution
"""

def visualize_rgb(tif_path, cut_off_value = 2000, show=False, save="tmp.png", force_process_all=False):
    # Hints to speed it up using overviews:
    # - https://rasterio.readthedocs.io/en/latest/topics/overviews.html
    # - https://gis.stackexchange.com/questions/353794/decimated-and-windowed-read-in-rasterio
    # Slow using rasterio ~ 5 images = 41.680561229994055s
    plot = plt.figure()
    
    # Open and read RGB bands
    src = rasterio.open("gs://" + tif_path) # Can I open fast in lower resolution?

    if not force_process_all:
        if src.width*src.height > 3451*4243: #5776040: # 3940,1466 still loaded, larger froze ...
            print("skipping too large~ ", src.width, src.height, src)
            return None, None
        if src.width*src.height < 500*500:
            print("skipping too small~ ", src.width, src.height, src)
            return None, None

    print("opening ~ ", src.width, src.height, src)

    image_rgb = src.read([4,3,2]) # src.read(window=window) would loot at a corner only ...
    red, green, blue = image_rgb

    # Threshold to deal with outliers
    red[red>cut_off_value] = cut_off_value
    blue[blue>cut_off_value] = cut_off_value
    green[green>cut_off_value] = cut_off_value

    # Scale bands
    red = (rasterio.plot.adjust_band(red, kind='linear')*255).astype(np.uint8)
    green = (rasterio.plot.adjust_band(green, kind='linear')*255).astype(np.uint8)
    blue = (rasterio.plot.adjust_band(blue, kind='linear')*255).astype(np.uint8)
    
    array = np.stack([red, green, blue], axis=0) # returns (3, 1497, 1698)
    resolution = array.shape

    rasterio.plot.show(array)
    plot.tight_layout()
    plt.axis('off')

    if show:
        plot.show() # only on non-vm machines
    if save:
        plot.savefig(save)
        
    return plot, resolution

def file_exists(fs, path):
    return fs.exists(path)

def visualize_folder(fs, folder, force_process_all=False):
    csv_with_links = load_csv_with_file_links(fs, folder)
    plots = []
    resolution = 0

    for index, row in csv_with_links.iterrows():
        exists = file_exists(fs, row.filename)
        print(index, row.filename, exists)
        
        if exists:
            plot, resolution = visualize_rgb(row.filename, force_process_all=force_process_all)
            if plot is None and resolution is None: # was too large, skipped ...
                continue
            plots.append(plot)

    return plots, csv_with_links, resolution


def init_wandb():
    wandb.login()


def wandb_show_folders(folders, dry_run = False, load_n_rows=15, wandb_table_name = "table_view_v4", force_process_all=False):
    # Note: check more media at https://docs.wandb.ai/guides/track/log
    #       tables at https://docs.wandb.ai/guides/data-vis/log-tables


    if not dry_run:
        init_wandb()
        #wandb.init(project='visualize_change_floods', entity='mlpayloads') 
        wandb.init(project="visualize_change_floods", config={})
        
    data_rows = []
    expected_images = 5
    columns= ["image_"+str(i) for i in range(expected_images)] + ["height", "width", "number_of_images", "folder"]
    try:
        for folder_i, folder in enumerate(folders):
            print(len(data_rows), "/", load_n_rows, "(folder", folder_i,"/444)") 
            if len(data_rows) == load_n_rows: break
            
            if folder_has_images(fs, folder):
                plots, csv_with_links, resolution = visualize_folder(fs, folder, force_process_all=force_process_all)

                number_of_images = len(plots)
                if number_of_images == 0:
                    plt.close('all')
                    continue

                if len(plots) > expected_images:
                    print("We have more than expected images in the folder ... use only the last", expected_images)
                    plots = plots[-expected_images:]

                # as a table:
                images = [wandb.Image(plot) for plot in plots]
                for i in range(expected_images - len(images)):
                    images.append( wandb.Image(np.zeros((100,100,3))) ) # missing images ...

                _, height, width = resolution
                single_row = images + [height, width, number_of_images, folder]
                data_rows.append(single_row)
                plt.close('all')

                # as an entry
                # now represent this as a row in wandb data
                """
                row_dict = {"folder":folder}
                for index, plot in enumerate(plots):
                    # this works, but adds every image as an independent entity
                    # I'd rather have rows ...
                    row_dict["image_"+str(index)] = wandb.Image(plot) 
                    # wandb.log({"example": wandb.Video("myvideo.mp4")})

                    #plot.savefig("image_"+str(index)+".png")
                
                plt.close('all')
                
                if dry_run:
                    print(row_dict)
                else:
                    wandb.log(row_dict)
                """

    except KeyboardInterrupt:
        print("Interupted with", len(data_rows), "samples ... will still try to save them...")

    # Logs only the final table (not row by row)
    print("debug~", data_rows)
    test_table = wandb.Table(data=data_rows, columns=columns)
    wandb.log({wandb_table_name: test_table})

    if not dry_run:
        wandb.finish()




if __name__=="__main__":
    fs = fsspec.filesystem("gs")

    """
    # On all folders:
    folders = fs.glob("gs://fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/*")
    np.random.shuffle(folders)
    subset = folders # ends after 20 rows anyways
    print(len(subset), subset[0:2])
    wandb_show_folders(subset, dry_run = False, load_n_rows=len(subset))

    """

    # Only on filtered, "well behaving" folders:
    #root_folder = "gs://fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/*/*.csv"
    root_folder = "gs://fdl-ml-payload/worldfloods_change_singleScene_500x500/EMSR260_02VIADANA_DEL_MONIT01_v2_observed_event_a/S2/*.csv"
    root_folder = "gs://fdl-ml-payload/worldfloods_change_singleScene_800x1000/*/S2/*.csv"
    root_folder = "gs://fdl-ml-payload/worldfloods_change_singleScene_2000x1500/*/S2/*.csv"
    root_folder = "gs://fdl-ml-payload/fire_change_singleScene/*/S2/*.csv"
    # all fire samples (may have a lot more than just 5 !)
    root_folder = "gs://fdl-ml-payload/fire_change/test_download_5/fire_change/S2/*/*.csv"

    
    root_folder = "gs://fdl-ml-payload/worldfloods_change_no_duplicates/train/*/S2/*.csv"
    force_process_all=True

    _, filtered_folders, _ = filter_file_list(fs, root_folder)
    print("Prefiltered", len(filtered_folders), "folders.")

    wandb_show_folders(filtered_folders, dry_run = False, load_n_rows=len(filtered_folders),force_process_all=force_process_all)


    # Timing loading of a single event:
    """
    start = timer()
    folder = "fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/EMSR258_06VLORE_DEL_v2_observed_event_a"
    visualize_folder(fs, folder)
    end = timer()
    time = (end - start)
    print("This run took "+str(time)+"s ("+str(time/60.0)+"min)")
    """
