import torch
from sklearn.decomposition import PCA
from model import AutoEncoder, load_autoencoder
from loaddata import loadCIFAR10
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def latent_pca(pca: PCA, data: np.ndarray) -> np.ndarray:
    return pca.transform(data)

# all=False uses mle estimator, otherwise forces all dimensions
def get_pca(autoencoder: AutoEncoder, testdata: DataLoader, device='cpu', all=False):
    data_vectors = list()
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    encoder = autoencoder.encoder
    encoder.eval()

    with torch.no_grad():
        for image, _ in testdata:
            image = image.to(device)
            compressed: torch.Tensor = encoder(image)
            flattened = compressed.mean((2, 3))
            data_vectors.append(flattened.to('cpu'))

    data_vectors = torch.cat(data_vectors).numpy()

    print(data_vectors.shape)

    ncomp = data_vectors.shape[1] if all else 'mle'

    pca = PCA(n_components=ncomp)
    pca.fit(data_vectors)
    return pca, data_vectors

# graphs only in 3d so use sparingly
def graph_pca(full_transformed: np.ndarray):
    num_components = full_transformed.shape[1]
    if num_components > 3:
        full_transformed = full_transformed[:, :3]
    elif num_components < 3:
        full_transformed = np.pad(full_transformed, ((0, 0), (0, 3 - num_components)))
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.scatter(full_transformed[:, 0], full_transformed[:, 1], full_transformed[:, 2])
    plt.show()

def pca_transform_truncated(pca: PCA, data: np.ndarray, ndim=None):
    if ndim is None:
        ndim = pca_dim_threshold(pca)
    return pca.transform(data)[:, :ndim]

def plot_variances(pca: PCA, semilog=False):
    if semilog:
        plt.semilogy(pca.explained_variance_)
    else:
        plt.plot(pca.explained_variance_)
    plt.show()

def pca_dim_threshold(pca: PCA, threshold=0.9) -> int:
    sum = 0.0
    count = 0
    for var in pca.explained_variance_ratio_:
        if sum >= threshold:
            break
        sum += var
        count += 1
    return count

if __name__ == '__main__':
    autoencoder = load_autoencoder('autoencoder.pth')
    trainloader, testloader = loadCIFAR10()
    pca, data = get_pca(autoencoder, testloader, all=False)
    transformed = latent_pca(pca, data)
    # print(pca_dim_threshold(pca))
    graph_pca(transformed)
    # plot_variances(pca)

