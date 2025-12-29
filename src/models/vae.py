import torch
import torch.nn as nn
import torch.nn.functional as F


class IsingVAE(nn.Module):
    """Variational Autoencoder for Ising model spin configurations."""
    
    def __init__(self, L: int, latent_dim: int = 2):
        super().__init__()
        self.L = L
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
        )
        
        # Compute feature dim
        with torch.no_grad():
            x = torch.zeros(1, 1, L, L)
            feat = self.encoder(x).shape[-1]
        
        # Latent space projection
        self.fc_mu = nn.Linear(feat, latent_dim)
        self.fc_logvar = nn.Linear(feat, latent_dim)
        
        # Decoder
        # Compute spatial dimensions after pooling
        spatial_dim = L // 4
        self.decoder_fc = nn.Linear(latent_dim, 128 * spatial_dim * spatial_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        self.spatial_dim = spatial_dim
    
    def encode(self, x):
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(-1, 128, self.spatial_dim, self.spatial_dim)
        return self.decoder(h)
    
    def forward(self, x):
        """Forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

