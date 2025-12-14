import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MockSegmentationDataset(Dataset):
    def __init__(self, length=10, image_size=(448, 448), num_classes=150):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random image
        image = torch.rand(3, *self.image_size)
        # Random segmentation mask
        mask = torch.randint(0, self.num_classes, self.image_size)
        return image, mask

class MockDetectionDataset(Dataset):
    def __init__(self, length=10, image_size=(448, 448), num_classes=80):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.rand(3, *self.image_size)
        # Mock detection target: heatmap for simplicity in this demo
        # In a real scenario, this would be bounding boxes
        target = torch.zeros(self.num_classes, 14, 14) # Downsampled grid
        # Add some random "objects"
        for _ in range(3):
            c = torch.randint(0, self.num_classes, (1,)).item()
            h = torch.randint(0, 14, (1,)).item()
            w = torch.randint(0, 14, (1,)).item()
            target[c, h, w] = 1.0
        return image, target

class MockDepthDataset(Dataset):
    def __init__(self, length=10, image_size=(448, 448)):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.rand(3, *self.image_size)
        # Mock depth map (0-1 range)
        depth = torch.rand(1, *self.image_size)
        return image, depth
