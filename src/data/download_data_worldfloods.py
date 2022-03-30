import traceback

import ee
import time
from google.cloud import storage
import os
from glob import glob
from typing import Optional, Callable, List, Tuple
from shapely.geometry import Polygon, mapping
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
import fsspec
import tempfile
import rasterio
import rasterio.warp
import subprocess
import numpy as np
import json
import math


BANDS_S2_NAMES = {
    "COPERNICUS/S2": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "QA60"],
    "COPERNICUS/S2_SR": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "SCL"]
}


def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    # https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    # https://stackoverflow.com/questions/40132542/get-a-cartesian-projection-accurate-around-a-lat-lng-pair/40140326#40140326
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = 'EPSG:326' + utm_band
        return epsg_code
    epsg_code = 'EPSG:327' + utm_band
    return epsg_code


def download_permanent_water(date, bounds):
    year = date.year
    # permananet water files are only available pre-2019
    if year >= 2019:
        year = 2019
    return ee.Image(f"JRC/GSW1_2/YearlyHistory/{year}").clip(bounds)


def get_collection(collection_name, date_start, date_end, bounds):
    collection = ee.ImageCollection(collection_name)
    collection_filtered = collection.filterDate(date_start, date_end) \
        .filterBounds(bounds)

    n_images = int(collection_filtered.size().getInfo())

    return collection_filtered, n_images


def get_s2_collection(date_start, date_end, bounds, collection_name="COPERNICUS/S2"):
    """
    Returns a daily mosaicked S2 collection over the specified dates and bounds. The collection includes
    the s2cloudless cloud probability as the last band of each image. Each image also has two metadata fields
    `'valids'` with the number of valid S2 pixels over the bounds and `'cloud_probability'` with the average of the
    cloud probability band (which has values from 0-100).

    Args:
        date_start:
        date_end:
        bounds:
        collection_name:

    Returns:

    """
    img_col_all, n_images_col = get_collection(collection_name, date_start, date_end, bounds)
    if n_images_col <= 0:
        print(f"Not images found for collection {collection_name} date start: {date_start} date end: {date_end}")
        return

    bands = BANDS_S2_NAMES[collection_name]
    img_col_all = img_col_all.select(bands)

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterBounds(bounds)
                        .filterDate(date_start, date_end))

    img_col_all = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': img_col_all,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    # Add s2cloudless as new band
    img_col_all = img_col_all.map(lambda x: x.addBands(ee.Image(x.get('s2cloudless')).select('probability')))

    daily_mosaic = collection_mosaic_day(img_col_all, bounds,
                                         fun_before_mosaic=None)

    # fun_before_mosaic=lambda img: img.toFloat().resample("bicubic")) # Bicubic resampling for 60m res bands?

    # Filter images with many invalids
    def _count_valid_clouds(img):
        mascara = img.mask()
        mascara = mascara.select(bands)
        mascara = mascara.reduce(ee.Reducer.allNonZero())
        dictio = mascara.reduceRegion(reducer=ee.Reducer.mean(), geometry=bounds,
                                      bestEffort=True, scale=10.)

        img = img.set("valids", dictio.get("all"))

        # Count clouds
        cloud_probability = img.select("probability")
        dictio = cloud_probability.reduceRegion(reducer=ee.Reducer.mean(), geometry=bounds,
                                                bestEffort=True, scale=10.)

        img = img.set("cloud_probability", dictio.get("probability"))

        return img

    daily_mosaic = daily_mosaic.map(_count_valid_clouds)

    return daily_mosaic


def collection_mosaic_day(imcol, region_of_interest, fun_before_mosaic=None):
    """
    Groups by solar day the images in the image collection.

    Args:
        imcol:
        region_of_interest:
        fun_before_mosaic:

    Returns:

    """
    # https://gis.stackexchange.com/questions/280156/mosaicking-a-image-collection-by-date-day-in-google-earth-engine
    imlist = imcol.toList(imcol.size())

    # longitude, latitude = region_of_interest.centroid().coordinates().getInfo()
    longitude = region_of_interest.centroid().coordinates().get(0)

    hours_add = ee.Number(longitude).multiply(12 / 180.)
    # solar_time = utc_time - hours_add

    unique_solar_dates = imlist.map(
        lambda im: ee.Image(im).date().advance(hours_add, "hour").format("YYYY-MM-dd")).distinct()

    def mosaic_date(solar_date_str):
        solar_date = ee.Date(solar_date_str)
        utc_date = solar_date.advance(hours_add.multiply(-1), "hour")

        ims_day = imcol.filterDate(utc_date, utc_date.advance(1, "day"))

        dates = ims_day.toList(ims_day.size()).map(lambda x: ee.Image(x).date().millis())
        median_date = dates.reduce(ee.Reducer.median())

        # im = ims_day.mosaic()
        if fun_before_mosaic is not None:
            ims_day = ims_day.map(fun_before_mosaic)

        im = ims_day.mosaic()
        return im.set({
            "system:time_start": median_date,
            "system:id": solar_date.format("YYYY-MM-dd"),
            "system:index": solar_date.format("YYYY-MM-dd")
        })

    mosaic_imlist = unique_solar_dates.map(mosaic_date)
    return ee.ImageCollection(mosaic_imlist)


PROPERTIES_DEFAULT = ["system:index", "system:time_start"]


def img_collection_to_feature_collection(img_col, properties=PROPERTIES_DEFAULT):
    properties = ee.List(properties)

    def extractFeatures(img):
        values = properties.map(lambda prop: img.get(prop))
        dictio = ee.Dictionary.fromLists(properties, values)
        return ee.Feature(img.geometry(), dictio)

    return ee.FeatureCollection(img_col.map(extractFeatures))


def findtask(description):
    task_list = ee.data.getTaskList()
    for t in task_list:
        if t["description"] == description:
            if (t["state"] == "READY") or (t["state"] == "RUNNING"):
                return True
    return False


def mayberun(filename, desc, function, export_task, overwrite=False, dry_run=False, verbose=1,
             bucket_name="worldfloods"):
    if bucket_name is not None:
        bucket = storage.Client().get_bucket(bucket_name)
        blobs_rasterized_geom = list(bucket.list_blobs(prefix=filename))

        if len(blobs_rasterized_geom) > 0:
            if overwrite:
                print("\tFile %s exists in the bucket. removing" % filename)
                for b in blobs_rasterized_geom:
                    b.delete()
            else:
                if verbose >= 2:
                    print("\tFile %s exists in the bucket, it will not be downloaded" % filename)
                return
    else:
        files = glob(f"{filename}*")
        if len(files) > 0:
            if overwrite:
                print("\tFile %s exists in the bucket. removing" % filename)
                for b in files:
                    os.remove(b)
            else:
                if verbose >= 2:
                    print(f"\tFile {filename} exists , it will not be downloaded")
                return

    if not dry_run and findtask(desc):
        if verbose >= 2:
            print("\ttask %s already running!" % desc)
        return

    if dry_run:
        print("\tDRY RUN: Downloading file %s" % filename)
        return

    try:
        image_to_download = function()

        if image_to_download is None:
            return

        print("\tDownloading file %s" % filename)

        task = export_task(image_to_download, fileNamePrefix=filename, description=desc)

        task.start()

        return task

    except Exception:
        traceback.print_exc()

    return


def export_task_image(bucket=Optional["worldfloods"], resolution_meters=10, file_dims=12_544,
                      maxPixels=5_000_000_000, crs='EPSG:4326', crsTransform=None) -> Callable:
    """
    function to export images in the WorldFloods format.

    Args:
        bucket:
        resolution_meters:
        file_dims:
        maxPixels:
        crs:
        crsTransform:

    Returns:

    """

    if bucket is not None:
        def export_task(image_to_download, fileNamePrefix, description):
            task = ee.batch.Export.image.toCloudStorage(image_to_download,
                                                        fileNamePrefix=fileNamePrefix,
                                                        description=description,
                                                        crs=crs.upper(),
                                                        crsTransform=crsTransform,
                                                        skipEmptyTiles=True,
                                                        bucket=bucket,
                                                        scale=resolution_meters,
                                                        formatOptions={"cloudOptimized": True},
                                                        fileDimensions=file_dims,
                                                        maxPixels=maxPixels)
            return task
    else:
        def export_task(image_to_download, fileNamePrefix, description):
            task = ee.batch.Export.image.toDrive(image_to_download,
                                                 fileNamePrefix=fileNamePrefix,
                                                 description=description,
                                                 crs=crs.upper(),
                                                 crsTransform=crsTransform,
                                                 skipEmptyTiles=True,
                                                 scale=resolution_meters,
                                                 formatOptions={"cloudOptimized": True},
                                                 fileDimensions=file_dims,
                                                 maxPixels=maxPixels)
            return task

    return export_task


def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]]


def wait_tasks(tasks: List[ee.batch.Task]) -> None:
    task_down = []
    for task in tasks:
        if task.active():
            task_down.append((task.status()["description"], task))

    task_error = 0
    while len(task_down) > 0:
        print("%d tasks running" % len(task_down))
        for _i, (t, task) in enumerate(list(task_down)):
            if task.active():
                continue
            if task.status()["state"] != "COMPLETED":
                print("Error in task {}:\n {}".format(t, task.status()))
                task_error += 1
            del task_down[_i]

        time.sleep(60)

    print("Tasks failed: %d" % task_error)


def download_s2(area_of_interest: Polygon, date_start_search: datetime, date_end_search: datetime, event_date: datetime,
                path_bucket: str, collection_name="COPERNICUS/S2_SR", crs='EPSG:4326', n_samples_before = 4,
                crsTransform=None,
                threshold_invalid=.75, threshold_clouds=.75,
                resolution_meters=10) -> Tuple[List[ee.batch.Task], Optional[pd.DataFrame]]:
    """
    Download time series of S2 images between search dates over the given area of interest. It saves the S2 images on
    path_bucket location. It only downloads images with less than threshold_invalid invalid pixels and with less than
    threshold_clouds cloudy pixels.

    Args:
        area_of_interest:
        date_start_search:
        date_end_search:
        path_bucket:
        collection_name:
        crs:
        crsTransform: Row-major list with the CRS transform
        threshold_invalid:
        threshold_clouds:
        resolution_meters:

    Returns:
        List of running tasks and dataframe with metadata of the S2 files.

    """

    assert path_bucket.startswith("gs://"), f"Path bucket: {path_bucket} must start with gs://"

    path_bucket_no_gs = path_bucket.replace("gs://", "")
    bucket_name = path_bucket_no_gs.split("/")[0]
    path_no_bucket_name = "/".join(path_bucket_no_gs.split("/")[1:])

    ee.Initialize()
    area_of_interest_geojson = mapping(area_of_interest)

    pol = ee.Geometry(area_of_interest_geojson)

    # Grab the S2 images
    img_col = get_s2_collection(date_start_search, date_end_search, pol,
                                collection_name=collection_name)
    if img_col is None:
        print(
            f"S2 images between {date_start_search.isoformat()} and {date_end_search.isoformat()} Total: 0")
        return [], None

    # Get info of the S2 images (convert to table)
    img_col_info = img_collection_to_feature_collection(img_col,
                                                        ["system:time_start", "valids",
                                                         "cloud_probability"])

    img_col_info_local = gpd.GeoDataFrame.from_features(img_col_info.getInfo())
    img_col_info_local["datetime"] = img_col_info_local["system:time_start"].apply(
        lambda x: datetime.utcfromtimestamp(x / 1000))
    img_col_info_local["cloud_probability"] /= 100
    img_col_info_local = img_col_info_local[["system:time_start", "valids", "cloud_probability", "datetime"]]
    img_col_info_local["index_image_collection"] = np.arange(img_col_info_local.shape[0])

    img_col_info_local["before_after_flag"] = np.zeros(img_col_info_local.shape[0])

    for row_i in range(img_col_info_local.shape[0]):
        difference = img_col_info_local["datetime"][row_i] - event_date
        #print("difference", difference, type(difference))
        if difference > timedelta(minutes=0.0):
            img_col_info_local.loc[row_i, "before_after_flag"] = 1.0

    n_images_col = img_col_info_local.shape[0]

    imgs_list = img_col.toList(n_images_col, 0)

    export_task_fun_img = export_task_image(
        bucket=bucket_name,
        crs=crs,
        crsTransform=crsTransform,
        resolution_meters=resolution_meters,
    )


    print(f"S2 images between {date_start_search.isoformat()} and {date_end_search.isoformat()} Total: {n_images_col}")

    image_col_after = img_col_info_local[img_col_info_local["before_after_flag"]==1]
    image_col_before = img_col_info_local[img_col_info_local["before_after_flag"]==0].iloc[:-2]

    filter_good = (image_col_before["cloud_probability"] <= threshold_clouds) & (
                image_col_before["valids"] > (1 - threshold_invalid))

    img_col_info_local_good = image_col_before[filter_good]
    #print("img_col_info_local_good", img_col_info_local_good)

    # Filtering out unwanted images
    # 1. select uniformly k=4 (let's say) samples as "before"
    n_before = len(image_col_before)
    n_after = len(image_col_after)

    if n_after < 1:
        print("All images after the event are bad")
        return [], img_col_info_local

    if n_before < n_samples_before:
        print("Not enough images before the event which are good (we have",n_before,"but need",n_samples_before,")")
        return [], img_col_info_local
    
    selected_good_images_before = img_col_info_local_good.tail(n_samples_before)

    # 2. select first sample after the event as "after"
    first_event_after = image_col_after.head(1)

    # Concatenate results
    selected_events = pd.concat([selected_good_images_before, first_event_after])
    selected_events = selected_events.reset_index(drop=True)
    print("selected_events", selected_events)

    tasks = []
    for good_images in selected_events.itertuples():
        #print("task => ",good_images)
        img_export = ee.Image(imgs_list.get(good_images.index_image_collection))
        img_export = img_export.select(BANDS_S2_NAMES[collection_name] + ["probability"]).toFloat().clip(pol)

        date = good_images.datetime.strftime('%Y-%m-%d')

        name_for_desc = os.path.basename(path_no_bucket_name)
        filename = os.path.join(path_no_bucket_name, date)
        desc = f"{name_for_desc}_{date}"
        task = mayberun(
            filename,
            desc,
            lambda: img_export,
            export_task_fun_img,
            overwrite=False,
            dry_run=False,
            bucket_name=bucket_name,
            verbose=2,
        )
        if task is not None:
            tasks.append(task)

    return tasks, selected_events


THRESHOLD_INVALIDS = .20
THRESHOLD_CLOUDS = .20
THRESHOLD_INVALIDS = .70
THRESHOLD_CLOUDS = .70

LOC_SAVE = "gs://fdl-ml-payload/worldfloods_change/"


def check_rerun(name_dest_csv, scene_id):
    """ Check if any S2 image is missing to trigger download """
    if not fs.exists(name_dest_csv):
        return True

    data = pd.read_csv(name_dest_csv)
    filter_good = (data["cloud_probability"] <= THRESHOLD_CLOUDS) & (data["valids"] > (1 - THRESHOLD_INVALIDS))
    data = data[filter_good]
    data["datetime"] = data["system:time_start"].apply(lambda x: datetime.utcfromtimestamp(x / 1000))
    for i in range(data.shape[0]):
        date = data['datetime'][i].strftime('%Y-%m-%d')
        filename = os.path.join(folder_dest, date + ".tif")
        if not fs.exists(filename):
            print(f"Missing files for product {scene_id}. Re-run")
            return True
    return False


if __name__ == "__main__":
    collection_name = "COPERNICUS/S2"
    collection_path = "gs://fdl-ml-payload/worldfloods_change_TestDownload3/"

    fs = fsspec.filesystem("gs")
    meta_files = fs.glob("gs://fdl-ml-payload/worldfloods_v1_0/*/meta/*")
    resolution_out_meters = 10

    tasks = []
    for _i, meta_file in enumerate(meta_files):
        print(f"{_i}/{len(meta_files)} processing {meta_file}")

        with fs.open(f"gs://{meta_file}") as fh:
            metadata = json.load(fh)
        date_image = datetime.fromisoformat(metadata['satellite date'][:-1])
        date_start_search = date_image + timedelta(days=-60)
        date_end_search = date_image + timedelta(days=10)

        # Name of the scene
        scene_id = os.path.splitext(os.path.basename(meta_file))[0]

        # S2 images will be stored in folder_dest path.
        # We will save a csv with the images queried and the available S2 images for that date
        train_test_val_string = meta_file.split("/")[-3]
        folder_dest = os.path.join(collection_path, train_test_val_string, "S2", scene_id)
        basename_csv = f"{date_start_search.strftime('%Y%m%d')}_{date_end_search.strftime('%Y%m%d')}.csv"
        name_dest_csv = os.path.join(collection_path, train_test_val_string, "S2", scene_id, basename_csv)

        if not check_rerun(name_dest_csv, scene_id):
            print(f"All data downloaded for product: {scene_id}")
            continue

        # Query S2 images for this location

        # we need the crs to download the images that we get from the corresponding Maxar tiff

        bounds = metadata["bounds"]
        crs = convert_wgs_to_utm((bounds[0]+bounds[2])/ 2., (bounds[1]+bounds[3])/ 2.)
        bounds_crs = rasterio.warp.transform_bounds({"init": "EPSG:4326"}, {"init": crs}, *bounds)
        crsTransform = [resolution_out_meters, 0 , min(bounds_crs[0], bounds_crs[2]),
                        0, -resolution_out_meters, max(bounds_crs[1], bounds_crs[3])]

        crsTransform = None

        pol_scene_id = Polygon(generate_polygon(bounds))
        tasks_iter, dataframe_images_s2 = download_s2(pol_scene_id, date_start_search=date_start_search,
                                                      date_end_search=date_end_search,
                                                      event_date=date_image,
                                                      crs=crs,crsTransform=crsTransform,
                                                      path_bucket=folder_dest,
                                                      threshold_invalid=THRESHOLD_INVALIDS,
                                                      threshold_clouds=THRESHOLD_CLOUDS,
                                                      collection_name=collection_name,
                                                      )

        if dataframe_images_s2 is None:
            continue

        # Create csv and copy to bucket
        with tempfile.NamedTemporaryFile(mode="w", dir=".", suffix=".csv", prefix=os.path.splitext(basename_csv)[0],
                                         delete=False, newline='') as fh:
            dataframe_images_s2.to_csv(fh, index=False)
            basename_csv_local = fh.name

        subprocess.run(["gsutil", "-m", "mv", basename_csv_local, name_dest_csv])
        tasks.extend(tasks_iter)

    wait_tasks(tasks)



