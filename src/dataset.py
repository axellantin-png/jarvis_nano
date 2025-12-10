import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, token_ids, block_size=128):
        """
        token_ids : liste ou tensor 1D de tous les tokens du corpus concaténés
        On découpe en fenêtres de taille block_size.
        """
        self.block_size = block_size
        self.data = token_ids

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
