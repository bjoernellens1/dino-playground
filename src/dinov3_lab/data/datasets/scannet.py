import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class ScanNetDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        """
        ScanNet Dataset loader for RGB-D.
        Expected structure:
            root/
                scans/
                    scene0000_00/
                        color/
                            0.jpg
                            ...
                        depth/
                            0.png
                            ...
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        self.scans_dir = os.path.join(root, "scans")
        if not os.path.exists(self.scans_dir):
             raise FileNotFoundError(f"Scans directory not found: {self.scans_dir}")
             
        self.samples = []
        # Walk through scans directory to find paired color/depth images
        # This can be slow for full ScanNet, so usually we'd use a split file.
        # For this implementation, we'll just list directories.
        
        scan_ids = sorted(os.listdir(self.scans_dir))
        # Simple split logic for demo: first 80% train, rest val
        split_idx = int(0.8 * len(scan_ids))
        if split == "train":
            scan_ids = scan_ids[:split_idx]
        else:
            scan_ids = scan_ids[split_idx:]
            
        for scan_id in scan_ids:
            scan_path = os.path.join(self.scans_dir, scan_id)
            color_dir = os.path.join(scan_path, "color")
            depth_dir = os.path.join(scan_path, "depth")
            
            if not os.path.exists(color_dir) or not os.path.exists(depth_dir):
                continue
                
            color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")])
            for f in color_files:
                depth_file = f.replace(".jpg", ".png")
                if os.path.exists(os.path.join(depth_dir, depth_file)):
                    self.samples.append((
                        os.path.join(color_dir, f),
                        os.path.join(depth_dir, depth_file)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        color_path, depth_path = self.samples[idx]
        
        image = Image.open(color_path).convert("RGB")
        depth = Image.open(depth_path) # 16-bit png usually
        
        return image, depth
