import numpy as np 
import pandas as pd 
from glob import glob 
import fsspec
from matplotlib import pyplot as plt 


CSV_PATH = "gs://fdl-ml-payload/worldfloods_change_TestDownload3/train/S2/*/*.csv"
fs = fsspec.filesystem("gs")
csv_files = fs.glob(CSV_PATH)

dfs = []
for fi in csv_files:
    print("Reading {}".format(fi))
    dfs.append(pd.read_csv("gs://"+fi))

total_dfs = pd.concat(dfs)

total_dfs["cloud_probability"].hist()