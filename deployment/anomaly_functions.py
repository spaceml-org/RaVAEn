import numpy as np
import torch
from typing import List, Any, Dict, Tuple
from torch import Tensor

def KL_divergence(mu1: Tensor, log_var1: Tensor, mu2: Tensor, log_var2: Tensor, reduce_axes: Tuple[int] = (-1,)):
    """ returns KL(D_2 || D_1) assuming Gaussian distributions with diagonal covariance matrices, and taking D_1 as reference 
    ----
    mu1, mu2, log_var1, log_var2: Tensors of sizes (..., Z) (e.g. (Z), (B, Z))
    
    """
    
    log_det = log_var1 - log_var2
    trace_cov = (-log_det).exp()
    mean_diff = (mu1 - mu2)**2 / log_var1.exp()
    return 0.5 * ((trace_cov + mean_diff + log_det).sum(reduce_axes) - mu1.shape[-1])


def twin_ae_change_score(model, x_1, x_2, verbose=False):
    if "SimpleAE" in str(model.__class__):
        latent_1 = model.encode(x_1) # batch, latent_dim
        #reconstruction_1 = model.decode(latent_1) # batch, channels, w, h

        latent_2 = model.encode(x_2) # batch, latent_dim
        #reconstruction_2 = model.decode(latent_2) # batch, channels, w, h

    else:
        assert False, "To be implemented!"

    if verbose:
        print("x_1", type(x_1), len(x_1), x_1[0].shape) # x 256 torch.Size([3, 32, 32]) 
        print("latent_1", type(latent_1), len(latent_1), latent_1[0].shape) # latent 256 torch.Size([256, 2, 2])

    # Converting from a volume of differences between inputs and reconstructions into a single channel map
    # For every pixel we can count mean accross all channels.
    #reconstruction_errors_1 = (x_1 - reconstruction_1)**2
    #reconstruction_errors_2 = (x_2 - reconstruction_2)**2
    #anomaly_map_1 = reconstruction_errors_1.mean(axis=1) # batch, w, h
    #anomaly_map_2 = reconstruction_errors_2.mean(axis=1) # batch, w, h
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = 1-cos(torch.flatten(latent_1, start_dim=1), torch.flatten(latent_2, start_dim=1))
    if verbose: print("distance", type(distance), len(distance), distance[0].shape)

    # convert to numpy
    distance = distance.detach().cpu().numpy()
    if verbose: print("distance", distance.shape)

    return distance



def twin_vae_change_score(model, x_1, x_2, verbose=False):
    if "SimpleVAE" in str(model.__class__):
        mu_1, log_var_1 = model.encode(x_1) # batch, latent_dim
        mu_2, log_var_2 = model.encode(x_2) # batch, latent_dim

    else:
        assert False, "To be implemented!"

    if verbose:
        print("x_1", type(x_1), len(x_1), x_1[0].shape) # x 256 torch.Size([3, 32, 32]) 
        print("mu_1", type(mu_1), len(mu_1), mu_1[0].shape) # 
        print("log_var_1", type(log_var_1), len(log_var_1), log_var_1[0].shape) # 

    distance = KL_divergence(mu_1, log_var_1, mu_2, log_var_2)
    #dst = wasserstein2(mu_1, log_var_1, mu_2, log_var_2)
    
    if verbose: print("distance", type(distance), len(distance), distance[0].shape)

    # convert to numpy
    distance = distance.detach().cpu().numpy()
    if verbose: print("distance", distance.shape)

    return distance