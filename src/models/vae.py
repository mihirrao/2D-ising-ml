import torch
import torch.nn as nn
import torch.nn.functional as F


class IsingVAE(nn.Module):
    def __init__(self, L: int, latent_dim: int = 2):
        super().__init__()
        self.L = L
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            x = torch.zeros(1, 1, L, L)
            feat = self.encoder(x).shape[-1]
        
        self.fc_mu = nn.Linear(feat, latent_dim)
        self.fc_logvar = nn.Linear(feat, latent_dim)
        
        spatial_dim = L // 4
        self.decoder_fc = nn.Linear(latent_dim, 128 * spatial_dim * spatial_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )
        self.spatial_dim = spatial_dim
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

