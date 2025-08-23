from typing import Tuple
import torch
from torch.utils.data import Dataset, random_split, DataLoader


class ToyNextIntDataset(Dataset):
    def __init__(self, N=100, repeats=10):
        super().__init__()
        self.N = N
        self.repeats = 10
    
    def __len__(self) -> int:
        return (self.N - 1) * self.repeats
    
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        if i < 0 or i >= len(self):
            raise IndexError()
        
        j = i % (self.N - 1)
        
        x = torch.tensor(j, dtype=torch.long)
        y = x + 1
        return x, y

def make_dataloaders(N, batch_size=16, shuffle=True) -> Tuple[DataLoader, DataLoader]:
    tds = ToyNextIntDataset(N=N)
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(tds, [0.8, 0.2], generator=generator)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)
    
    return train_dl, test_dl

