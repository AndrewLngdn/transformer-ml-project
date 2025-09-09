import torch
from typing import Tuple, Dict, List
from torch.utils.data import Dataset


def build_char_vocab(s: str) -> Tuple[List[str], Dict[str, int]]:
    itos = sorted(set(s))
    
    stoi = {}
    
    for i, char in enumerate(itos):
        stoi[char] = i
        
    return itos, stoi


def encode(s: str, stoi) -> torch.Tensor:
    ints = []
    for c in s:
        try:
            ints.append(stoi[c])
        except KeyError as e:
            print(f"{c} isn't in vocabulary")
            raise e
            
    return torch.tensor(ints, dtype=torch.int64)


def decode(t: torch.Tensor, itos) -> str:
    ints = t.tolist()
    chars = []
    
    for int in ints:
        chars.append(itos[int])
    
    return "".join(chars)


def split_ids(chars_tensor: torch.Tensor, frac=0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    train_len = int(len(chars_tensor) * frac)
    
    train_tensor = chars_tensor[:train_len]
    val_tensor = chars_tensor[train_len:]
    
    return train_tensor, val_tensor


class CharDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, T=16):
        super().__init__()
        self.tokens = tokens
        self.T = T

    def __len__(self) -> int:
        """
        len() is the number of windows that the dataset can provide. 
        """
        total_window_len = self.T + 1
        return len(self.tokens) - total_window_len + 1
    
    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        if i < 0 or i >= len(self):
            raise IndexError()
        
        return self.tokens[i:i+self.T], self.tokens[i+1:i+1+self.T]

if __name__ == "__main__":
    print("Run tests with: uv run pytest tests/test_char_data.py")