# models/simple_encoder.py
import torch
import torch.nn as nn

class SimpleConvEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(embed_dim, in_channels, 1, padding=0),
            nn.ReLU()
        )
        # self.relu = nn.ReLU()
    def forward(self, x):
        return self.encoder(x) + x
