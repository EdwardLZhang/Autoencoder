import torch
import torch.nn as nn
from model import AutoEncoder
import matplotlib.pyplot as plt
from loaddata import loadCIFAR10

def visualize_autoencoding(autoencoder, testdata, num_images=5):
    images, _ = next(iter(testdata))
    images = images.to(device)

    with torch.no_grad():
        reconstructed = autoencoder(images)

    fig, axes = plt.subplots(num_images, 2)

    for i in range(min(num_images, len(images))):
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0))
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed[i].cpu().permute(1, 2, 0))
        axes[i, 1].axis('off')

    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the model
    autoencoder: AutoEncoder = AutoEncoder()
    state = torch.load('autoencoder.pth', map_location='cpu')
    autoencoder.load_state_dict(state)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # load the test data

    traindata, testdata = loadCIFAR10()

    visualize_autoencoding(autoencoder, testdata)