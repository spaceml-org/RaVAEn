# Deployment readme:

This part of the code will have a simpler extracted version of the model to run on Pynq. In order to do this, as few specialized libraries will be used. The code will also contain large amount of monitoring scripts.

## Code flow proposal:

The barebones functionality (with the tracked parameters) is:

```
1a. load location - RAM, IO
1b. load tiles - RAM, IO

2. load model - RAM, IO

3. run model on tiles - time

4. save result - IO
```

## Prerequisites:

Set up your PYNQ board accoring to https://github.com/manoharvhr/PYNQ-Torch. We used the preinstalled image from https://github.com/manoharvhr/PYNQ-Torch/releases/tag/v1.0.

Python libraries:

```
torch==1.2.0
numpy==1.16.4
```

Then start with:

```
python3.6 run_v0.py
```
