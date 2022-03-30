# Train models later evaluated in papers
# Medium model ~ VAE 128 with 0 skip connections

train () {
    rm -rf /data/cache
    rm -rf $HOME/cache/
    python3 -m scripts.train_model +dataset=alpha_multiscene ++dataset.root_folder="/data/local/worldfloods_change_no_duplicates/train" \
         +normalisation=log_scale +channels=high_res +training=$1 +module=$2 +project=train_VAE_128medium +name="${3}" \
         module.model_cls_args.latent_dim=128 module.model_cls_args.extra_depth_on_scale=0

}

training=simple_vae
module=deeper_vae
name=VAE_128medium

train $training $module $name


