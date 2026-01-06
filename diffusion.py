from pydiffmap import diffusion_map as dm
import numpy as np
from model import load_autoencoder
from loaddata import loadCIFAR10
from pca import get_pca, pca_transform_truncated

if __name__ == '__main__':
    autoencoder = load_autoencoder('autoencoder.pth')
    trainloader, testloader = loadCIFAR10()
    pca, data = get_pca(autoencoder, testloader, all=False)
    transformed = pca_transform_truncated(pca, data)
    dmap = dm.DiffusionMap.from_sklearn(
        n_evecs=3
        # this is based on the visualization from pca.py
        # n_evecs-1 is the approximate dimensionality of the data

    )