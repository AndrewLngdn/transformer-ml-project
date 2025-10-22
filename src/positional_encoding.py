import torch
from torch import nn


class LazySinCosPositionalEncoding(nn.Module):
    def __init__(self, dropout, embed_dim, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.P = None # Init on device of input
        
    def forward(self, X):
        B, L, D = X.shape
        
        if self.P is None:
            grid = torch.cartesian_prod(torch.arange(self.max_len), torch.arange(self.embed_dim)).reshape(self.max_len, self.embed_dim, 2)
            freq = grid[:, :, 1] // 2  # floor divide to pair sin/cos dims
            self.P = torch.where(
                grid[:, :, 1] % 2 == 0,
                torch.sin(grid[:, :, 0] / 10000 ** (2 * freq / self.embed_dim)),
                torch.cos(grid[:, :, 0] / 10000 ** (2 * freq / self.embed_dim))
            ).to(X.device)
            
        X = self.dropout(X + self.P[:L, :])
        
        return X
        
        
        