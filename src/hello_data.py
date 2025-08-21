from typing import Tuple
import torch
from torch.utils.data import Dataset


class ToyNextIntDataset(Dataset):
    def __init__(self, N=100):
        super().__init__()
        self.N = N
    
    def __len__(self) -> int:
        return self.N - 1
    
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        if i == len(self):
            raise IndexError()
        
        x = torch.tensor(i, dtype=torch.long)
        y = x + 1
        return x, y