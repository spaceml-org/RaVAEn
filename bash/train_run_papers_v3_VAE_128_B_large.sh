# Train models later evaluated in papers
# Large model ~ VAE 128

train () {
    rm -rf /data/cache
    rm -rf $HOME/cache/
    python3 -m scripts.train_model +dataset=alpha_multiscene ++dataset.root_folder="/data/local/worldfloods_change_no_duplicates/train" \
         +normalisation=log_scale +channels=high_res +training=$1 +module=$2 +project=train_VAE_128 +name="${3}" module.model_cls_args.latent_dim=128


}

training=simple_vae
module=deeper_vae
name=VAE_128

train $training $module $name


