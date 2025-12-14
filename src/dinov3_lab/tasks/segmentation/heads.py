import torch
import torch.nn as nn
from typing import Tuple

class LinearSegmentationHead(nn.Module):
    """
    A simple linear probe for segmentation.
    Projects dense features to class logits.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) dense features
        Returns:
            logits: (B, num_classes, H, W)
        """
        return self.head(x)
