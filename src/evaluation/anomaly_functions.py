"""
This module contains functions to use GRX and VAE modules to caluate anomaly scores
"""
import numpy as np


def vae_anomaly_function(model, x, invalid_mask):
    x = x.cuda()

    if "SimpleAE" in str(model.model.__class__):
        reconstruction = model.forward(x)  # batch, channels, w, h
    else:
        assert False, \
            "To be impented, VAE might return more values in forward()"

    print("x", type(x), len(x), x[0].shape)
    print("reconstruction", type(reconstruction),
          len(reconstruction), reconstruction[0].shape)

    # Converting from a volume of differences between inputs and
    # reconstructions into a single channel map
    # For every pixel we can count mean accross all channels.
    reconstruction_errors = (x - reconstruction)**2
    anomaly_maps = reconstruction_errors.mean(axis=1)  # batch, w, h

    # convert to numpy
    anomaly_maps = anomaly_maps.detach().cpu().numpy()
    invalid_mask = invalid_mask.detach().cpu().numpy()
    print("anomaly_maps", anomaly_maps.shape)
    print("invalid_mask", invalid_mask.shape)

    masked_anomaly_maps = np.ma.masked_where(invalid_mask, anomaly_maps)
    # sum anomalies along all axis except the batch axis. Then normalise by
    # number of pixels in each image
    anomaly_scores = masked_anomaly_maps.sum(axis=tuple(n for n in range(1, len(masked_anomaly_maps.shape))))  # batch
    anomaly_scores = anomaly_scores.filled(fill_value=0) / np.product(masked_anomaly_maps.shape[1:])  # batch

    return np.log(anomaly_maps), anomaly_scores


def grx_anomaly_function(model, x, invalid_mask):
    x = x.detach().cpu().numpy()
    x = x.transpose((1, 0, 2, 3))  # channels, batch, w, h
    anomaly_maps = model.score(x)  # batch, w, h

    invalid_mask = invalid_mask.detach().cpu().numpy()
    masked_anomaly_maps = np.ma.masked_where(invalid_mask, anomaly_maps)

    # sum anomalies along all axis except the batch axis.
    # Then normalise by number of pixels in each image
    anomaly_scores = masked_anomaly_maps.sum(axis=tuple(n for n in range(1, len(masked_anomaly_maps.shape))))  # batch
    anomaly_scores = anomaly_scores.filled(fill_value=0) / np.product(masked_anomaly_maps.shape[1:])  # batch

    return np.log(anomaly_maps), anomaly_scores


#### For change detection:
def vae_anomaly_function_with_latents(model, x, invalid_mask, latents_keep_tensors=False):
    x = x.cuda()
    if "SimpleAE" in str(model.model.__class__):
        reconstruction = model.model.forward(x) # batch, channels, w, h
        latent = model.model.encode(x) # batch, latent_dim
        if not latents_keep_tensors: latent = latent.detach().cpu().numpy()
        latent = latent.reshape((len(latent), -1))

    elif "SimpleVAE" in str(model.model.__class__):
        reconstruction, mu, log_var = model.model.forward(x) # [0] batch, channels, w, h, [1,2] batch, latent_dim (1024)
        if not latents_keep_tensors: mu = mu.detach().cpu().numpy()
        if not latents_keep_tensors: log_var = log_var.detach().cpu().numpy()
        latent = [mu, log_var]
    else:
        assert False, "To be implemented, VAE might return more values in forward()"
        # reconstruction = model.forward(x)[0]

    #print("x", type(x), len(x), x[0].shape) # x 256 torch.Size([3, 32, 32]) 
    #print("reconstruction", type(reconstruction), len(reconstruction), reconstruction[0].shape) # reconstruction 256 torch.Size([3, 32, 32])
    #print("latent", type(latent), len(latent), latent[0].shape) # latent 256 torch.Size([256, 2, 2])

    # Converting from a volume of differences between inputs and reconstructions into a single channel map
    # For every pixel we can count mean accross all channels.
    reconstruction_errors = (x - reconstruction)**2
    anomaly_maps = reconstruction_errors.mean(axis=1) # batch, w, h
    
    # convert to numpy
    anomaly_maps = anomaly_maps.detach().cpu().numpy()
    invalid_mask = invalid_mask.detach().cpu().numpy()
    print("anomaly_maps", anomaly_maps.shape)
    print("invalid_mask", invalid_mask.shape)
    #    latent.reshape((len(latent), np.product(latent.shape[1:])))
    #print("latent", latent.shape)

    masked_anomaly_maps = np.ma.masked_where(invalid_mask, anomaly_maps)
    # sum anomalies along all axis except the batch axis. Then normalise by number of pixels in each image
    anomaly_scores = masked_anomaly_maps.sum(axis=tuple(n for n in range(1, len(masked_anomaly_maps.shape)))) # batch
    anomaly_scores = anomaly_scores.filled(fill_value=0) / np.product(masked_anomaly_maps.shape[1:]) # batch

    return np.log(anomaly_maps), anomaly_scores, latent


def vae_anomaly_only_latents(model, x, latents_keep_tensors=False):
    x = x.cuda()

    if "SimpleAE" in str(model.model.__class__):
        latent = model.model.encode(x) # batch, latent_dim
        if not latents_keep_tensors: latent = latent.detach().cpu().numpy()
        latent = latent.reshape((len(latent), -1))

    elif "SimpleVAE" in str(model.model.__class__):
        _, mu, log_var = model.model.forward(x) # [0] batch, channels, w, h, [1,2] batch, latent_dim (1024)
        if not latents_keep_tensors: mu = mu.detach().cpu().numpy()
        if not latents_keep_tensors: log_var = log_var.detach().cpu().numpy()
        latent = [mu, log_var]
    else:
        assert False, "To be implemented"

    return latent