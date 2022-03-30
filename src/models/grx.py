import numpy as np


def pixelwise_image_diff(image_before,
                         image_after,
                         threshold=3,
                         reduction=None):
    """
    Calculate pixelwise image difference
    Parameters:
    ----------
    image: numpy array
        input image in (x,y,bands) order
    threshold: int
        number of standard deviations to use for threshold
    reduction: function
        function to reduce mask over bands.
    """

    # Calculate differences
    diff = np.abs(image_before-image_after)
    std = diff.std(axis=(0, 1), keepdims=True)
    mask = diff > std
    # Reduce over bands
    if reduction:
        mask = np.apply_along_axis(reduction, -1, mask)

    return mask


class GRX:
    """
    GRX method which is 'trained' over all the batches of a single location
    image and then evaluated for each tile separately.
    """

    def __init__(self):
        self.cov = None
        self.mu = None
        self.channels = None
        self.pixels_seen_for_mean = 0
        self.pixels_seen_for_cov = 0
        self.inv_cov = None

    def partial_fit_mean(self, image):
        channels = image.shape[0]
        if self.channels is not None:
            assert channels == self.channels
        else:
            self.channels = channels
        N = np.product(image.shape[1:])
        image = image.reshape(channels, N)
        mu = image.mean(axis=1)  # channels
        if self.mu is None:
            self.mu = mu
        else:
            self.mu = (self.mu * self.pixels_seen_for_mean + mu * N)
            self.mu /= (self.pixels_seen_for_mean + N)

        self.pixels_seen_for_mean += N
        # If we update the means then the covariance becomes invalid
        self.cov = None
        self.pixels_seen_for_cov = 0

    def partial_fit_cov(self, image):
        """
        Parameters:
        -----------
        image: np.array
            image of shape (channels, ...)

        Returns:
        --------
        None
        """
        channels = image.shape[0]
        if self.channels is not None:
            assert channels == self.channels
        else:
            self.channels = channels
        N = np.product(image.shape[1:])
        image = image.reshape(channels, N)

        # Covariance without de-meaning the data from the batch
        # we instead use the accumulated mean value
        image_dev = image - self.mu[:, np.newaxis]
        cov = (image_dev @ image_dev.T) / N  # channels x channels

        if self.cov is None:
            self.cov = cov
        else:
            self.cov = (self.cov * self.pixels_seen_for_cov + cov * N)
            self.cov /= (self.pixels_seen_for_cov + N)

        self.pixels_seen_for_cov += N
        self.inv_cov = None

    def score(self, image):
        """
        Parameters:
        -----------
        image: np.array
            array of shape (channels, ...)

        Returns:
        --------
        anomaly_scores: np.array
            array of shape (...)
        """
        shape = image.shape
        channels = shape[0]
        N = np.product(shape[1:])
        image = image.reshape(channels, N)

        if self.inv_cov is None:
            # using the accumulated covariance (channels x channels)
            self.inv_cov = np.linalg.pinv(self.cov)

        # and accumulated mean (channels x N)
        image_dev = image - self.mu.reshape(channels, 1)

        distance = \
            np.apply_along_axis(lambda x: x @ self.inv_cov @ x, 1, image_dev.T)

        return distance.reshape(shape[1:])

    def preprocess_data(self, input):
        # For now this is mimicking the preprocessing that we do in VAE ...
        # Might help with some linear algebra instabilities, might not ...
        # we have to test it!
        input /= 3.
        input[input > 1.] = 1.
        input[input < -1.] = -1.

        return input

    def report(self):
        print("GRX.cov", self.cov.shape, self.cov)
        print("GRX.mu", self.mu.shape, self.mu)
        print("GRX.channels", self.channels)
        print("GRX.pixels_seen_for_mean", self.pixels_seen_for_mean)
        print("GRX.pixels_seen_for_cov", self.pixels_seen_for_cov)


if __name__ == "__main__":
    pass
