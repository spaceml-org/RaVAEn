from torch import Tensor

from .base_ae import BaseAE


class BaseVAE(BaseAE):
    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
        
    def _visualise_step(self, batch):
        result = self.forward(batch) # [reconstruction, mu, log_var]
        # Just select the reconstruction
        result = result[0]
        rec_error = (batch - result).abs()

        return batch[:, self.visualisation_channels], \
            result[:, self.visualisation_channels], \
            rec_error.max(1)[0]
