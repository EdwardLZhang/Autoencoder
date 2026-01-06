import torch
import torch.nn as nn
from typing import Union

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.f(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.f(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        compressed = self.encoder(x)
        return self.decoder(compressed)


def load_autoencoder(path: str, device=None) -> Union[AutoEncoder, None]:
    print(f'Loading autoencoder from {path}')
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    autoencoder: AutoEncoder = AutoEncoder()
    try:
        state = torch.load(path, map_location='cpu')
        autoencoder.load_state_dict(state)
        autoencoder = autoencoder.to(device)
        print('Autoencoder loaded successfully')
    except Exception:
        autoencoder = None
        print('Failed to load autoencoder')
    return autoencoder
