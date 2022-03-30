from PIL.Image import init
import torch
from torch.nn.functional import cosine_similarity, interpolate
from src.evaluation.vae_metrics import KL_divergence, wasserstein2
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def predictions2image(predictions, grid_shape, tile_size):
    # Converts predicitons array into an numpy array image
    channels = 1
    image = np.zeros((channels, grid_shape[1]*tile_size, grid_shape[0]*tile_size), dtype=np.float32)
    index = 0
    for i in range(grid_shape[1]):
        for j in range(grid_shape[0]):
            tile = predictions[index] * np.ones((channels, tile_size, tile_size))
            image[:, i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
            index += 1

    return image


def expand_embedding(emb, tile_size):
    need_reshape = len(emb.shape) == len(tile_size)
    if need_reshape:
        emb = emb[None, None, ...]
    embig = interpolate(emb, scale_factor=tile_size, mode='nearest').detach()
    if need_reshape:
        embig = embig[0, 0]
    return embig

# TODO: defs change_masks pixel2tile ~ different ways to go from pixel change map gt into a tile score
#                                      for example number of pixel change / number of all change -> monotonously growing score
#                                      or more than a threshold of pixel change -> change label


### TODO:
## Qualitative:
# - AE cosine distance
# - VAE cosine distance (means)
# - VAE KL divergence
# - VAE wasserstein distance


## Quantitative
# - same as all above (!), then compare with metric





class MethodBase:
    returns = None
    def __call__(self, **kwargs):
        raise NotImplementedError

    @property
    def _name(self):
        """Default name in table is just class name. 
        Can be overwritten with class attribute.
        """
        return self.__class__.__name__

    @property
    def name(self):
        return self._name
    

class ImageMethodBase(MethodBase):
    returns = 'image'

    def __init__(self, tile_size=(1,1), overlap=(0, 0), memory_size=1, aggregate_method='mean') -> None:
        super().__init__()

        self.tile_size = (tile_size[0] - overlap[0], tile_size[1] - overlap[1])
        self.memory_size = memory_size
        self.aggregate_method = aggregate_method

    @property
    def name(self):
        return f"{self._name} | memory {self.memory_size} | {self.tile_size[0]}x{self.tile_size[1]} - {self.aggregate_method}"

    def aggregate(self, image):
        res = torch.zeros(image.shape[0] // self.tile_size[0], image.shape[1] // self.tile_size[1]).to(image.device)
        h, w = res.shape

        for i in range(h):
            for j in range(w):
                image_slice = image[ 
                    i*self.tile_size[0]:(i+1) * self.tile_size[0], 
                    j*self.tile_size[1]:(j+1) * self.tile_size[1]
                ]
                if self.aggregate_method == 'mean':
                    agg_image_slice = image_slice.mean()
                elif self.aggregate_method == 'min':
                    agg_image_slice = image_slice.min()
                elif self.aggregate_method == 'max':
                    agg_image_slice = image_slice.max()
                else:
                    raise NotImplementedError

                res[i, j] = agg_image_slice
        return res
    

class StatMethodBase(MethodBase):
    returns = 'value'

    
class DiffPixels(ImageMethodBase):
    _name = "diff_pixels"
    
    def __call__(self, data_now, data_win, **kwargs):
        if data_win.shape[0] == 0:
            return torch.zeros(1, 1, 1)

        data_win = data_win[-self.memory_size:].cuda()
        data_now = data_now[0].cuda()

        res = torch.cat([torch.linalg.norm(data_now - dw, dim=0)[None, ] for dw in data_win])
        res = res.min(0)[0]
        
        res = self.aggregate(res)
        image = expand_embedding(res, self.tile_size)
        return image.detach().cpu()


class CosPixels(ImageMethodBase):
    _name = 'cos_pixel'

    def __call__(self, data_now, data_win, **kwargs):
        if data_win.shape[0] < self.memory_size:
            return torch.zeros(1, 1, 1)

        data_win = data_win[-self.memory_size:].cuda()
        data_now = data_now[0].cuda()

        res = torch.cat([cosine_similarity(data_now, dw, 0)[None, ] for dw in data_win])
        res = res.max(0)[0]
        
        res = self.aggregate((1 - res) / 2.)
        image = expand_embedding(res, self.tile_size)
        return image.detach().cpu()


class CosEmbeddingImage(ImageMethodBase):
    # Direct cosine distance between embeddings. Note: for VAE we can do better, but for AE is fine.

    _name = 'cos_emb'

    def __call__(self, embs_now, embs_win, is_vae=False, **kwargs):
        if embs_win.shape[0] == 0:
            return torch.zeros(1, 1, 1)

        if is_vae: # VAE has concatenated mu and logvar
            embs_now = embs_now[:, :embs_now.shape[1]//2]
            embs_win = embs_win[:, :embs_win.shape[1]//2]

        embs_win = embs_win[-self.memory_size:].cuda()
        embs_now = embs_now[0].cuda()

        res = torch.cat([cosine_similarity(embs_now, ew, 0)[None, ] for ew in embs_win])
        res = res.max(0)[0]
        res = (1 - res) / 2.
        # grid_shape = [embs_now.shape[-2], embs_now.shape[-1]] # ~ might need to be fixed
        # image = predictions2image(res, grid_shape, self.tile_size)
        image = expand_embedding(res, self.tile_size)
        return image.detach().cpu()


class DiffEmbeddingImage(ImageMethodBase):
    # Direct eucledian distance between embeddings. Note: for VAE we can do better, but for AE is fine.

    _name = 'diff_emb'

    def __call__(self, embs_now, embs_win, is_vae=False, **kwargs):
        if embs_win.shape[0] == 0:
            return torch.zeros(1, 1, 1)

        if is_vae: # VAE has concatenated mu and logvar
            embs_now = embs_now[:, :embs_now.shape[1]//2]
            embs_win = embs_win[:, :embs_win.shape[1]//2]

        embs_win = embs_win[-self.memory_size:].cuda()
        embs_now = embs_now[0].cuda()

        res = torch.cat([torch.linalg.norm(embs_now - ew, dim=0)[None, ] for ew in embs_win])
        res = res.min(0)[0]
        image = expand_embedding(res, self.tile_size)
        return image.detach().cpu()


class VAEImageMethodBase(ImageMethodBase):
    for_vae = True # only valid for VAEs not AEs
    def __init__(self, tile_size=(1,1), overlap=None, memory_size=None, aggregate_method=None) -> None:
        super().__init__()

        self.tile_size = (tile_size[0] - overlap[0], tile_size[1] - overlap[1])
        self.memory_size = memory_size
        self.aggregate_method = aggregate_method

    
    @staticmethod
    def split_embedding_and_var(embvar):
        mu = embvar[:, :embvar.shape[1]//2]
        logvar = embvar[:, embvar.shape[1]//2:]
        return mu, logvar
        

class KLDivEmbeddingImage(VAEImageMethodBase):
    # KL divergence between two encodings. Returns an image visualization.

    def __call__(self, embs_now, embs_win, **kwargs):
        if embs_win.shape[0] == 0:
            return torch.zeros(1, 1, 1)
        mu_win, log_var_win = self.split_embedding_and_var(embs_win)
        mu_now, log_var_now = self.split_embedding_and_var(embs_now)
        # mu_now[0] has shape [C, W, H] going into this func
        res = KL_divergence(mu_win[-1], log_var_win[-1], mu_now[0], log_var_now[0], reduce_axis=0)
        # res has shape [W,H]
        image = expand_embedding(res, self.tile_size)
        return image


class WasserEmbeddingImage(VAEImageMethodBase):
    # Wasserstein divergence between two encodings. Returns an image visualization.
    def __call__(self, embs_now, embs_win, **kwargs):
        if embs_win.shape[0] == 0:
            return torch.zeros(1, 1, 1)

        mu_win, log_var_win = self.split_embedding_and_var(embs_win)
        mu_now, log_var_now = self.split_embedding_and_var(embs_now)
        res = wasserstein2(mu_win[-1], log_var_win[-1], mu_now[0], log_var_now[0], reduce_axis=0)
        image = expand_embedding(res, self.tile_size)
        return image
    
    
class PrecisionRecallBase(StatMethodBase):
    
    def precision_recall(self, true_changes, pred_change_scores):
        # convert to numpy arrays and mask invalid
        # After these lines the changes and scores are 1D
        invalid_masks  = [c.numpy()==2 for c in true_changes]
        true_changes = np.concatenate(
            [c.numpy()[~m] for m,c in zip(invalid_masks, true_changes)],
            axis=0
        )
        pred_change_scores = np.concatenate(
            [c.numpy()[~m] for m,c in zip(invalid_masks, pred_change_scores)],
            axis=0
        )
        
        precision, recall, thresholds = precision_recall_curve(
            true_changes,
            pred_change_scores
        )
        return precision, recall, thresholds

    
class AreaUnderPrecisionRecall(PrecisionRecallBase):
    name = "area under precision recall curve"
    def __call__(self, true_changes, pred_change_scores):
        precision, recall, thresholds = self.precision_recall(true_changes, pred_change_scores)
        area_under_precision_curve = auc(recall, precision)
        return area_under_precision_curve

    
class PrecisionAt100Recall(PrecisionRecallBase):
    name = "precision at 100% recall"
    def __call__(self, true_changes, pred_change_scores):
        precision, recall, thresholds = self.precision_recall(true_changes, pred_change_scores)    
        precision_at_100_recall = precision[0]
        return precision_at_100_recall