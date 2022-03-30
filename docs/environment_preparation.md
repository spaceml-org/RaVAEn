# Data and environment preparation

## Repository and environment setup
To clone the repository of the project, run:
```
git clone git@gitlab.com:frontierdevelopmentlab/fdl-europe-2021-ml-payload/change-detection.git
```
In this repository, a `git flow` approach is used, where the `master` branch is
supposed to be working, `develop` to contain the cutting-edge code and the
single features to be owned by their own branches.

To set up the python environment, you can run:
```
make requirements
```
which will use the `env.yaml` file with `conda` to install the necessary
packages and to activate the right environment.

## Bucket mount and config setup
To mount your data, if you are on Google Cloud, you can simply mount a bucket
through the command:
```
gcsfuse --implicit-dirs my-bucket /path/to/mount/point
```

Once the data is mounted, some care has to be put to let the code know where
the data and folders are.
In particular the `config/config.yaml` file is used to link the right folders
for saving results and cached datamodules.

### Data folders
Particular care is to be taken for the data, depending ont he dataset of use.
In any case, the datamodule `ParsedDataModule` expect the events of interest
to be placed in a single folder, which will be its `root_folder`.
Specific datasets need a specific folder structure.
