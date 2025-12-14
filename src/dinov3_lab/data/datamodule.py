from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

class COCODataModule(BaseDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__(batch_size, num_workers)
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None):
        # TODO: Implement COCO loading logic
        pass

    def train_dataloader(self):
        # Placeholder
        return DataLoader([], batch_size=self.batch_size)

class ADE20KDataModule(BaseDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__(batch_size, num_workers)
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None):
        # TODO: Implement ADE20K loading logic
        pass

    def train_dataloader(self):
        # Placeholder
        return DataLoader([], batch_size=self.batch_size)

class ScanNetDataModule(BaseDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__(batch_size, num_workers)
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None):
        # TODO: Implement ScanNet loading logic
        pass

    def train_dataloader(self):
        # Placeholder
        return DataLoader([], batch_size=self.batch_size)
