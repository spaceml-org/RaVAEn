from typing import List, Any, Dict, Tuple

import numpy as np
import torch

from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

def KL_divergence(mu1: Tensor, log_var1: Tensor, mu2: Tensor, log_var2: Tensor, reduce_axis: int = -1):
    """ returns KL(D_2 || D_1) assuming Gaussian distributions with diagonal covariance matrices, and taking D_1 as reference 
    ----
    mu1, mu2, log_var1, log_var2: Tensors of sizes (..., Z) (e.g. (Z), (B, Z))
    
    """
    
    log_det = log_var1 - log_var2
    trace_cov = (-log_det).exp()
    mean_diff = (mu1 - mu2)**2 / log_var1.exp()
    return 0.5 * ((trace_cov + mean_diff + log_det).sum(reduce_axis) - mu1.shape[reduce_axis])

def KL_divergence_std(mu1: Tensor, log_var1: Tensor, mu2: Tensor, log_var2: Tensor):

    """ returns KL(D_2 || D_1) assuming Gaussian distributions with diagonal covariance matrices, and taking D_1 as reference 

    ----
    mu1, mu2, log_var1, log_var2: 1D Tensors of sizes (Z)
    
    """

    d1 = MultivariateNormal(mu1, covariance_matrix=torch.diag(log_var1.exp()))
    d2 = MultivariateNormal(mu2, covariance_matrix=torch.diag(log_var2.exp()))

    return kl_divergence(d2, d1)


def wasserstein2(mu1: Tensor, log_var1: Tensor, mu2: Tensor, log_var2: Tensor, reduce_axis: int = -1):

    return ((mu1 - mu2)**2 + (log_var1.exp()**0.5 - log_var2.exp()**0.5)**2).sum(reduce_axis)   

if __name__ == "__main__":

    from tqdm import tqdm

    mu1 = mu2 = torch.ones([5])
    log_var1 = log_var2 = torch.zeros([5])

    assert wasserstein2(mu1, log_var1, mu2, log_var2) == 0
    assert KL_divergence_std(mu1, log_var1, mu2, log_var2) == 0

    mu1 = torch.ones([5])
    mu2 = torch.tensor([1., 1.3, 1., 1., 1.])
    log_var1 = log_var2 = torch.zeros([5])

    assert torch.isclose(KL_divergence_std(mu1, log_var1, mu2, log_var2), KL_divergence(mu1, log_var1, mu2, log_var2))
    assert torch.isclose(wasserstein2(mu1, log_var1, mu2, log_var2), torch.tensor(0.3**2))

    count = 0
    for i in tqdm(range(100)):
        mu1 = 2 * torch.rand((10, )) - 1
        mu2 = 2 * torch.rand((10, )) - 1
        log_var1 = 2 * torch.rand((10, )) - 1
        log_var2 = 2 * torch.rand((10, )) - 1

        if torch.isclose(KL_divergence_std(mu1, log_var1, mu2, log_var2), KL_divergence(mu1, log_var1, mu2, log_var2)):
            count += 1
        # print(KL_divergence_std(mu1, log_var1, mu2, log_var2), KL_divergence(mu1, log_var1, mu2, log_var2))
    print(count)

    mu1 = mu2 = torch.ones([1, 5])
    log_var1 = log_var2 = torch.zeros([1, 5])

    assert torch.allclose(wasserstein2(mu1, log_var1, mu2, log_var2), torch.zeros([1, 5]))
    assert wasserstein2(mu1, log_var1, mu2, log_var2, reduce_axes=(0, 1)) == 0
    assert torch.allclose(wasserstein2(mu1, log_var1, mu2, log_var2, reduce_axes=(1,)), torch.zeros([5])) 

    # assert torch.allclose(KL_divergence_std(mu1, log_var1, mu2, log_var2), torch.zeros([1, 5]))