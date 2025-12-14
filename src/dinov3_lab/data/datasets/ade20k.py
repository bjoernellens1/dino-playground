from torch.utils.data import Dataset

class ADE20KDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.root = root
        self.split = split

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError
