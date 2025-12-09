import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .cad_dataset import CADDataset

class ImageCADDataset(CADDataset):
    def __init__(self, phase, config, feature_root):
        # Initialize parent CADDataset (handles vector and split)
        super(ImageCADDataset, self).__init__(phase, config)
        
        self.feature_root = feature_root

    def __getitem__(self, index):
        # Get vector data from parent
        data = super(ImageCADDataset, self).__getitem__(index)
        data_id = data['id']
        
        # Construct feature path
        # Structure: feature_root/0000/00000007.npy
        subfolder = data_id[:4] 
        feat_path = os.path.join(self.feature_root, subfolder, f"{data_id}.npy")
        
        try:
            # Load features: (4, 384)
            features = np.load(feat_path)
            features = torch.from_numpy(features).float()
        except (FileNotFoundError, OSError):
            # Handle missing features
            features = torch.zeros(4, 384)
            
        data['features'] = features
        
        return data
