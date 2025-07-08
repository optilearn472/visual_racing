import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # [1, 64, 96]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [32, 32, 48]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [64, 16, 24]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [64, 8, 12]
            nn.Flatten(),
            nn.Linear(64 * 8 * 12, 128),
            nn.ELU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 64)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
            )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 64 * 8 * 12),
            nn.ELU(),
            nn.Unflatten(1, (64, 8, 12)),
            # [64, 8, 12]
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [64, 16, 24]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # [32, 32, 48]
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        # print(h.shape)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar