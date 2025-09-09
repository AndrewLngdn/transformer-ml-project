import torch
from torch import nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, embedding_dim=128):
        super().__init__()
        
        self.embs = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
    
    def forward(self, X):
        embeddings = self.embs(X)
        output, (h_n, c_n) = self.LSTM(embeddings)
        vocab_logits = self.out(output)
        
        return vocab_logits
    


class CharLSTMTie(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        
        self.embs = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.LSTM = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.vocab_bias = nn.Parameter(torch.zeros((vocab_size)))
    
    def forward(self, X):
        embeddings = self.embs(X)
        output, (h_n, c_n) = self.LSTM(embeddings)
        
        # output_shape B, T, H
        # embs: V, H
        # want B, T, V
        # embs.T: H, V
        
        vocab_logits = output @ self.embs.weight.T + self.vocab_bias
        
        return vocab_logits
        # return vocab_logits
    
    