import torch
import torch.nn as nn

class IsingCNN(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            x = torch.zeros(1, 1, L, L)
            feat = self.net(x).shape[-1]
        self.head = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        z = self.net(x)
        return self.head(z)