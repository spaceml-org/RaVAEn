import numpy as np


def tiles2image(predicted_distances, grid_shape, overlap=0, tile_size = 32):
    # predicted_distances shape of ~ N
    channels = 1
    image = np.zeros((channels, grid_shape[1]*tile_size, grid_shape[0]*tile_size), dtype=np.float32)
    index = 0
    for i in range(grid_shape[1]):
        for j in range(grid_shape[0]):
            tile = predicted_distances[index] * np.ones((channels, tile_size, tile_size))
            image[:, i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
            index += 1
    return image


def save_plot(image, name):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.imshow(image)
    plt.colorbar()
    plt.savefig(name+".png")
    plt.close()
