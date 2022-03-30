import fsspec
import pandas as pd
import numpy as np 
from tqdm import tqdm


def load_csv_with_file_links(fs, folder):
    """
    Loads folder structure (original csv file and tif images) and adds a new column with filepaths for each row.
    Useful for later filtration.
    """
    files = fs.glob(folder+"/*")

    images = [f for f in files if f[-4:] == ".tif"]
    csv = [f for f in files if f[-4:] == ".csv"][0]
    metadata = pd.read_csv("gs://"+csv)
    
    # Each file in the folder is named <date>.tif
    metadata["filename"] = metadata["datetime"].apply(lambda x: folder+"/"+x.split(" ")[0]+".tif")

    return metadata

def filter_file_list(fs, root_folder, BEFORE_THR_INVALIDS = .25, AFTER_THR_INVALIDS = .25, 
                     BEFORE_THR_CLOUDS = .25, AFTER_THR_CLOUDS = .25, subset=None):
    """
    Apply different selection criteria for images before and after event.
    *_THR_INVALIDS 0.25 ~ allow only 25% invalid pixels (we want high valids) => checks for > 1-thr
    *_THR_CLOUDS 0.25 ~ allow up to 25% cloudy pixels (we want low clouds) => checks for < thr

    If an after event image is rejected, also ignore the before event images.
    Also returns folders which contain at least some of good images.
    """

    # Get a list of all csv files
    csv_files = fs.glob(root_folder)

    dfs = []
    filtered_folders = []

    # Make a dataframe with all csvs including filenames
    if subset is not None:
        csv_files = csv_files[:subset]
    for csv_path in tqdm(csv_files):
        # Add file paths to csv
        folder_path = "/".join(csv_path.split("/")[:-1])
        csv = load_csv_with_file_links(fs, folder_path)

        events_after = csv[csv["before_after_flag"]==1]
        if len(events_after) != 1:
            continue

        # check conditions on images after events
        condition = np.logical_and((events_after.valids > (1 - AFTER_THR_INVALIDS)),(events_after.cloud_probability < AFTER_THR_CLOUDS))

        # If after event image doesn't meet thresholds, drop all images of event
        if not condition.item():
            continue
        # Otherwise, check the conditions on images before the events
        else: 
            events_before = csv[csv["before_after_flag"]==0]
            condition = np.logical_and((events_before.valids > (1 - BEFORE_THR_INVALIDS)),(events_before.cloud_probability < BEFORE_THR_CLOUDS))

            """ debug for posterity
            for index, row in events_before.iterrows():
                if True: #row.index_image_collection == 17:
                    print(index, row)
                    print("cond:", condition[index])
                    print("valid?", row.valids > (1 - BEFORE_THR_INVALIDS))
                    print("cloud?", row.cloud_probability < (BEFORE_THR_CLOUDS))
            """

            dfs.append(pd.concat([events_before[condition], events_after]))
            filtered_folders.append(folder_path)

    #metadata = [df.filename]
    merged_df = pd.concat(dfs).reset_index(drop=True)
    list_of_dfs = dfs
    return merged_df, filtered_folders, list_of_dfs

if __name__=="__main__":
    fs = fsspec.filesystem("gs")
    #folder = "gs://fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/EMSR258_06VLORE_DEL_v2_observed_event_a"
    #load_csv_with_file_links(fs, folder)

    root_folder = "gs://fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/*/*.csv"
    metadata_df, filtered_folders, _ = filter_file_list(fs, root_folder) #, .6,.6,.6,.6)
    
    print("Prefiltered", len(filtered_folders), "folders.")
