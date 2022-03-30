# Runs evaluation on manually selected model checkpoint
#   (note: on different architecture layouts might need some extra args, ideally the same as the used training script)

eval_all_disasters () {
    for event in floods fires hurricanes landslides
    do
        rm -rf /data/cache
        rm -rf $HOME/cache/
        python3 -m scripts.evaluate_model \
                +dataset=floods_evaluation \
                ++dataset.root_folder="/data/local/validation_data_final/$event" \
                +training=$1 \
                +normalisation=log_scale \
                +channels=high_res \
                +module=$2\
                +checkpoint=$3 \
                +project="eval_paper_VAE_128" \
                +evaluation=$4 \
                ++evaluation.plot_sequences=$plot_sequences \
                +name="${5}_${event}" +dataset.test_overlap=[0,0] module.model_cls_args.latent_dim=128


    done
}

plot_sequences=true

###############
# VAEs
###############
evaluation=vae_paper
training=simple_vae
###############

checkpoint=/data/results/ex02_nips_vitek_trainings/B_train_VAE_128/mjtjgybo/checkpoints/epoch_00-step_29653.ckpt
module=deeper_vae
name=B_VAE_128_default_mjtjgybo_epoch0
eval_all_disasters $training $module $checkpoint $evaluation $name

checkpoint=/data/results/ex02_nips_vitek_trainings/B_train_VAE_128/sw5vnewy/checkpoints/epoch_00-step_29653.ckpt
module=deeper_vae
name=B_VAE_128_default_sw5vnewy_epoch0
eval_all_disasters $training $module $checkpoint $evaluation $name
