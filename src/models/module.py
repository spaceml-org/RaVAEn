import torch
from torch import optim
from src.utils import load_obj

import pytorch_lightning as pl


class Module(pl.LightningModule):
    def __init__(self, cfg: dict, train_cfg: dict) -> None:
        super().__init__()

        # get transformers args if available
        da_trans_cls = train_cfg.pop("da_trans_cls", "src.data.transformations.NoTransformer")
        da_trans_args = train_cfg.pop("da_trans_args", list())
        eval_trans_cls = train_cfg.pop("eval_trans_cls", "src.data.transformations.NoTransformer")
        eval_trans_args = train_cfg.pop("eval_trans_args", list())

        self.__dict__.update(cfg)
        self.__dict__.update(train_cfg)
        self.save_hyperparameters(cfg)

        model_cls = load_obj(self.model_class)
        self.model = \
            model_cls(input_shape=self.input_shape, **self.model_cls_args)
        
        # optional data augmentations to foster invariances
        self.da_transformer = \
                load_obj(da_trans_cls)(*da_trans_args)
        
        # optional transformation at evaluation (e.g. cropping tile to get right size)
        self.eval_transformer = \
                load_obj(eval_trans_cls)(*eval_trans_args)


        if hasattr(self.model, '_visualise_step'):
            self._visualise_step = \
                lambda batch: self.model._visualise_step(self.da_transformer(batch[0]))
            self._visualisation_labels = self.model._visualisation_labels

    def forward(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(batch, **kwargs)

    def log_losses(self, loss, where):
        for k in loss.keys():
            self.log(f'{where}/{k}', loss[k], on_epoch=True, logger=True)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        batch = batch[0]
        batch_size = batch.shape[0]
        
        batch_eval = self.eval_transformer(batch) # eval model with transformed data
        batch_da = self.da_transformer(batch) # apply model on data augmented batch
        results = self.forward(batch_da)

        train_loss = self.model.loss_function(batch_eval,
                                              results,
                                              M_N=batch_size/self.len_train_ds,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_losses(train_loss, 'train')

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        batch = batch[0]
        batch_size = batch.shape[0]

        batch_da = self.da_transformer(batch) # apply model on data augmented batch
        batch_eval = self.eval_transformer(batch) # eval model with transformed data
        results = self.forward(batch_da)
        
        val_loss = self.model.loss_function(batch_eval,
                                            results,
                                            M_N=batch_size / self.len_val_ds,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_losses(val_loss, 'valid')

        return val_loss

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        optims.append(optimizer)

        if hasattr(self, 'scheduler_gamma'):
            scheduler = \
                optim.lr_scheduler.ExponentialLR(optims[0],
                                                 gamma=self.scheduler_gamma)
            scheds.append(scheduler)

        if hasattr(self, 'lr2'):
            optimizer2 = \
                optim.Adam(getattr(self.model, self.submodel).parameters(),
                           lr=self.lr2)
            optims.append(optimizer2)

        # Check if another scheduler is required for the second optimizer
        if hasattr(self, 'scheduler_gamma2'):
            scheduler2 = \
                optim.lr_scheduler.ExponentialLR(optims[1],
                                                 gamma=self.scheduler_gamma2)
            scheds.append(scheduler2)

        return optims, scheds
