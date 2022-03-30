import torch
import wandb
from pytorch_lightning.callbacks import Callback


class VisualisationCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if not hasattr(pl_module, '_visualise_step'):
            return
        columns = ["index"] + pl_module._visualisation_labels

        dataset = trainer.val_dataloaders[0].dataset
        collate_fn = trainer.val_dataloaders[0].collate_fn

        image_grid = self.sample_from_vis_dataset(dataset,
                                                  pl_module, 5, collate_fn)

        test_table = wandb.Table(data=image_grid, columns=columns)
        trainer.logger.experiment.log(
            {f"Recontructions: ep{trainer.current_epoch}step{trainer.global_step}": test_table})

    def sample_from_vis_dataset(self,
                                dataset,
                                model,
                                n_images,
                                collate_fn=None):

        sample_images = []
        for idx in range(n_images):
            idx = torch.randint(0, len(dataset) - 1, (1, 1))

            sample = dataset[idx.item()]
            sample = [torch.tensor(s) for s in sample]
            if collate_fn:
                sample = collate_fn([sample])
            else:
                sample = [s[None, ] for s in sample]
            sample = [s.to(model.device) for s in sample]

            images = self.image_grid_generate(model, sample)

            row = [idx] + [wandb.Image(im) for im in images]
            sample_images.append(row)

        return sample_images

    def image_grid_generate(self, model, batch):
        results = model._visualise_step(batch)
        sample_images = [torch.nan_to_num(im).squeeze(0) for im in results]
        return sample_images
