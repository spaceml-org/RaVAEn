# Training and evaluation

## Training:
Trainign of a model is done through the `script/train_model.py` script, as in
the following command:
```
python3 -m scripts.train_model +dataset=alpha_singlescene \
                               +normalisation=log_scale \
                               +channels=all \
                               +training=simple_ae \
                               +module=simple_ae \
                               +project=ae_fullgrid
```

In this example, the model defined in `config/module/simple_ae.yaml` is trained
on the dataset defined in `config/dataset/alpha_singlescene.yaml`, `config/channels/all.yaml`
and `config/normalisation/log_scale.yaml` for the project `ae_fullgrid`, which
matches the `wandb` project name.

The trained model is going to be saved as a checkpoint in the result folder
defined in `config/config.yaml`.

To train a model with data augmentations, one can specify as training config the file `config/training/da.yaml` and append the chosen transformation config, e.g. `config/transform/random.yaml` for applying random band shifts to the inputs and a center crop to the outputs/targets. 
The full command:
```
python3 -m scripts.train_model +dataset=alpha_singlescene \
                               +normalisation=log_scale \
                               +channels=all \
                               +training=da \
                               +module=simple_ae \
                               +project=da_ae_fullgrid \
                               +transform=random
```

## Evaluation:
Once a model is trained, you can use the `scripts/evaluate_model.py` script to
evaluate it.
The model takes additional config parameters: `checkpoint` is the position of the
checkpoint of the model you want to evaluate and `evaluation` is a config file
where the list of  metrics are defined.

```
python3 -m scripts.evaluate_model \
    +dataset=floods_evaluation \
    +training=simple_ae \
    +normalisation=log_scale \
    +channels=rgb \
    +module=simple_ae_with_linear \
    +checkpoint=/data/ml_payload/results/vae_fullgrid/w1b4uo19/checkpoints/epoch_34-step_28139.ckpt \
    +project=eval_reboot \ 
    +evaluation=ae_base \
    #+name=whatever_name_you_want
```
