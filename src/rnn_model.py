from torch import nn


class NextCharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=64, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.RNN = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, dropout=dropout)
        self.out_proj = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
    def forward(self, X):
        embeddings = self.embedding(X)
        rnn_out, h_n = self.RNN(embeddings)
        vocab_logits = self.out_proj(rnn_out)
        return vocab_logits
    

