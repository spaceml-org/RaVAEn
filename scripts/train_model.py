from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import hydra

from src.utils import load_obj, deepconvert
from src.data.datamodule import ParsedDataModule
from src.callbacks.visualisation_callback import VisualisationCallback


@hydra.main(config_path='../config', config_name='config.yaml')
def main(cfg):

    seed_everything(42, workers=True)

    cfg = deepconvert(cfg)

    data_module = ParsedDataModule.load_or_create(cfg['dataset'],
                                                  cfg['cache_dir'])

    cfg['module']['len_train_ds'] = data_module.len_train_ds
    cfg['module']['len_val_ds'] = data_module.len_val_ds
    cfg['module']['len_test_ds'] = data_module.len_test_ds

    cfg['module']['input_shape'] = data_module.sample_shape_train_ds.to_tuple()[0]

    cfg_train = cfg['training']
    module = load_obj(cfg['module']['class'])(cfg['module'], cfg_train)

    log_name = cfg['module']['class'] + '/' + cfg['project']
    logger = loggers.WandbLogger(save_dir=cfg['log_dir'], name=log_name,
                                 project=cfg['project'], entity=cfg['entity'])

    callbacks = [
        VisualisationCallback(),
        LearningRateMonitor(),
        ModelCheckpoint(
            save_last=True,
            save_top_k=-1,  # -1 keeps all, # << 0 keeps only last ....
            filename='epoch_{epoch:02d}-step_{step}',
            auto_insert_metric_name=False)
    ]

    plugins = []
    if cfg_train.get('distr_backend') == 'ddp':
        plugins.append(DDPPlugin(find_unused_parameters=False))

    trainer = Trainer(
        deterministic=True,
        gpus=cfg_train['gpus'],
        logger=logger,
        callbacks=callbacks,
        plugins=plugins,
        profiler='simple',
        max_epochs=cfg_train['epochs'],
        accumulate_grad_batches=cfg_train['grad_batches'],
        accelerator=cfg_train.get('distr_backend'),
        precision=16 if cfg_train['use_amp'] else 32,
        auto_scale_batch_size=cfg_train.get('auto_batch_size'),
        auto_lr_find=cfg_train.get('auto_lr', False),
        check_val_every_n_epoch=cfg_train.get('check_val_every_n_epoch', 10),
        reload_dataloaders_every_epoch=False,
        fast_dev_run=cfg_train['fast_dev_run'],
        resume_from_checkpoint=cfg_train.get('from_checkpoint'),
        #check_val_every_n_epoch=cfg_train.get('check_val_every_n_epoch', 1) # line repeated?
    )

    trainer.tune(module, datamodule=data_module)

    trainer.fit(module, data_module)


if __name__ == '__main__':
    main()
