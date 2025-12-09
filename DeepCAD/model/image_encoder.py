import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=256, feature_dim=384):
        super(ImageEncoder, self).__init__()
        
        # Backbone is removed as we use pre-computed features
        self.feature_dim = feature_dim
        
        # 4 views * 384 dim = 1536
        self.fusion_dim = 4 * self.feature_dim
        
        # Projection Head
        # Maps fused features to 256-dim latent space
        # Ending with Tanh to match the Autoencoder's latent space range [-1, 1]
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
            nn.Tanh()
        )

    def forward(self, features):
        """
        Args:
            features: (B, 4, 384)
        Returns:
            z: (B, 256)
        """
        B, N, D = features.shape
        
        # Flatten views: (B, 4*384) = (B, 1536)
        features_fused = features.view(B, -1)
        
        # Project to latent space
        z = self.head(features_fused)
        
        return z
