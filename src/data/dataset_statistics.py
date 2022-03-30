#
# Used to query information about dataset statistics
#

import fsspec
from matplotlib import pyplot as plt 
import rasterio
import numpy as np 
from glob import glob 
from timeit import default_timer as timer


def get_pixel_counts_from_label(src):
    label_values = src.read()
    print(label_values.shape)
    unique, counts = np.unique(label_values, return_counts=True)  # 0 no change, 1 change, 2 cloud

    print(unique, counts)
    #import pdb; pdb.set_trace()

    nochange_pixels_count = counts[0]
    change_pixels_count = counts[1]
    cloud_pixels_count = 0
    if 2 in unique:
        cloud_pixels_count = counts[2]
    all_pixels = src.width * src.height
    all_pixels_check = nochange_pixels_count + change_pixels_count + cloud_pixels_count
    assert all_pixels == all_pixels_check

    return change_pixels_count, nochange_pixels_count, cloud_pixels_count

    print("all pixels:", all_pixels, "check", all_pixels_check)
    print("ch | no ch | cloud:", change_pixels_count, nochange_pixels_count, cloud_pixels_count)

    change_to_no_change = change_pixels_count / nochange_pixels_count
    print("change to no change ratio:", change_to_no_change)

if __name__=="__main__":
    fs = fsspec.filesystem("gs")
    
    datasets = [ "fires", "floods", "hurricanes", "landslides", "oilspills", "volcanos" ]    
    #datasets = [ "floods" ]

    count_pixels = True

    for dataset in datasets:
        root_folder = "gs://fdl-ml-payload/validation/validation_data_final/"+dataset+"/*"

        folders = fs.glob(root_folder)
        print(len(folders), folders)

        total_km2 = 0
        pixels_change = 0
        pixels_nochange = 0
        pixels_cloud = 0
        pixels_all = 0

        for folder in folders:
            images_path_string = "gs://" + folder + "/S2/*.tif"
            image_files = fs.glob(images_path_string)

            labels_path_string = "gs://" + folder + "/changes/*.tif"
            labels_files = fs.glob(labels_path_string)
            #print("Folder", folder, "has", len(image_files), "images and", len(labels_files), "label")

            # inspect the label dimensions:
            src = rasterio.open("gs://" + labels_files[0])
            #print("labels ~ ", src.width, src.height)

            width_km = float(src.width) / 100
            height_km = float(src.height) / 100
            
            area_km2 = width_km * height_km

            #print("area ~ ", len(image_files), "*", area_km2, "=", len(image_files)*area_km2, "km2")

            total_km2 += area_km2 #len(image_files)*area_km2

            if count_pixels:
                changes, nochanges, clouds = get_pixel_counts_from_label(src)
                pixels_change += changes
                pixels_nochange += nochanges
                pixels_cloud += clouds
                pixels_all += changes + nochanges# + clouds # ~ changes / changes + nochanges ~ ignoring clouds?


        print("===")
        print(dataset," > In total this set contains:", total_km2, "km2")
        print("Change pixels out of all used pixels:", (pixels_change / pixels_all) * 100, "%")
        print("Total change:", pixels_change, "|", "Total no-change:", pixels_nochange, "|", "Total clouds:", pixels_cloud)
        print("")
        print("")