from torch import nn

class HelloModel(nn.Module):
    def __init__(self, vocab_size=128, embedding_dim=128):
        super().__init__()
        
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        
        self.out = nn.Linear(embedding_dim, vocab_size)
        
    
    def forward(self, X):
        embeddings = self.emb(X)
        vocab_logits = self.out(embeddings)
        
        return vocab_logits