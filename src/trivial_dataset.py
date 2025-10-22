import torch
from torch.utils.data import Dataset


class TrivialDataset(Dataset):
    def __init__(self, vocab_size=5, seq_len=6, dataset_len=10, seq_len_min=1):
        self.seq_len = seq_len
        self.seq_len_min = seq_len_min
        self.dataset_len = dataset_len
        self.vocab_size = vocab_size
        self.PAD = 0
        
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        if index < 0 or index >= self.dataset_len:
            raise IndexError()

        src = torch.arange(index + 1, index + self.seq_len + 1, dtype=torch.long) % (self.vocab_size - 1) + 1
        tgt = torch.arange(index + 2, index + self.seq_len + 2, dtype=torch.long) % (self.vocab_size - 1) + 1
        new_seq_len = max((index + self.seq_len // 2) % self.seq_len, self.seq_len_min)
        src[new_seq_len:] = 0
        
        tgt_len = new_seq_len + 2
        tgt[tgt_len:] = 0
        return src, tgt
    


if __name__ == "__main__":
    tds = TrivialDataset(dataset_len=8, seq_len=64, vocab_size=5)
    
    for i, (src, tgt) in enumerate(tds):
        print()
        print(f"{i=} {src=}")
        print(f"{i=} {tgt=}")

    # i=0 src=tensor([1, 2, 3, 4, 5, 6, 0, 0])
    # i=0 tgt=tensor([2, 3, 4, 5, 6, 7, 0, 0])

    # i=1 src=tensor([2, 3, 4, 5, 6, 7, 0, 0])
    # i=1 tgt=tensor([3, 4, 5, 6, 7, 8, 0, 0])

    # i=2 src=tensor([3, 4, 5, 6, 7, 8, 0, 0])
    # i=2 tgt=tensor([4, 5, 6, 7, 8, 9, 0, 0])

    # i=3 src=tensor([4, 5, 6, 7, 8, 9, 0, 0])
    # i=3 tgt=tensor([ 5,  6,  7,  8,  9, 10,  0,  0])