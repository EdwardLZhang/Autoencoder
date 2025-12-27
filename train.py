from loaddata import loadCIFAR10
from model import AutoEncoder
import torch
import torch.nn as nn

def train_model(model: nn.Module, lossf: nn.Module, optimizer: torch.optim.Optimizer, trainloader, epochs=20, device='cpu'):
    for i in range(epochs):
        model.train()
        total_loss = 0
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = lossf(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    autoencoder = AutoEncoder()
    autoencoder.to(device)

    trainloader, testloader = loadCIFAR10()

    lossf = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters())
    train_model(autoencoder, lossf, optimizer, trainloader, device=device)
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    torch.save(autoencoder.encoder.state_dict(), 'encoder.pth')
    torch.save(autoencoder.decoder.state_dict(), 'decoder.pth')

