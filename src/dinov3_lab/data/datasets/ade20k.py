import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class ADE20KDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        """
        ADE20K Dataset loader.
        Expected structure:
            root/
                images/
                    training/
                    validation/
                annotations/
                    training/
                    validation/
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        split_dir = "training" if split == "train" else "validation"
        self.image_dir = os.path.join(root, "images", split_dir)
        self.mask_dir = os.path.join(root, "annotations", split_dir)
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
            
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".jpg", ".png")
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            # Note: Transforms need to handle both image and mask carefully
            # For simplicity in this demo, we assume transform handles tuple or we apply manually
            # But standard torchvision transforms don't handle both. 
            # We will return raw PIL images if no transform, or assume transform is a callable taking both.
            # Or simpler: just return PIL and let the training loop handle conversion/resizing.
            pass
            
        return image, mask
