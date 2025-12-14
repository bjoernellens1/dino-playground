import os
from PIL import Image
from torch.utils.data import Dataset
import torch

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

class COCODataset(Dataset):
    def __init__(self, root: str, split: str = "train2017", transform=None):
        """
        COCO Dataset loader.
        Expected structure:
            root/
                train2017/
                val2017/
                annotations/
                    instances_train2017.json
                    instances_val2017.json
        """
        if COCO is None:
            raise ImportError("pycocotools is not installed. Please install it to use COCODataset.")

        self.root = root
        self.split = split
        self.transform = transform
        
        self.image_dir = os.path.join(root, split)
        self.ann_file = os.path.join(root, "annotations", f"instances_{split}.json")
        
        if not os.path.exists(self.ann_file):
             raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        self.coco = COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.image_dir, path)
        
        # Robust loading: skip if file missing
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}. Skipping...")
            return self.__getitem__((idx + 1) % len(self))
            
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Failed to load image at {img_path}: {e}. Skipping...")
            return self.__getitem__((idx + 1) % len(self))
        
        # Convert coco_target to a format suitable for the model
        # For this playground, we might return the raw list of dicts 
        # or convert to boxes/labels.
        # Let's return raw image and target for the loop to handle collation
        
        return image, coco_target
