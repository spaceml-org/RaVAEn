# --- Might need to sudo bash run this script
# This script downloads the main datasets for our experiments 
# from the google bucket

mkdir -p /data/local

# for alpha_multiscene 
gsutil -m cp -r gs://fdl-ml-payload/worldfloods_change_no_duplicates /data/local/.

# for alpha_singlescene
gsutil -m cp -r gs://fdl-ml-payload/worldfloods_change_trainSingleScene_1800x2600_with_val /data/local.

# for floods_evaluation
gsutil -m cp -r gs://fdl-ml-payload/validation/validation_data_final /data/local/.

chmod -R 777 /data/local
