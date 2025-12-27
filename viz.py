import torch
import torch.nn as nn
from model import AutoEncoder
import matplotlib.pyplot as plt
from loaddata import loadCIFAR10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model
autoencoder: AutoEncoder = AutoEncoder()
state = torch.load('autoencoder.pth', map_location='cpu')
autoencoder.load_state_dict(state)
autoencoder = autoencoder.to(device)
autoencoder.eval()

# load the test data

traindata, testdata = loadCIFAR10()

images, _ = next(iter(testdata))
images = images.to(device)

fig, axes = plt.subplots(5, 2)

