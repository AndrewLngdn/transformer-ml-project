from torch import nn

from module_utils import init_seq2seq

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X, *args):
        raise NotImplementedError


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        # init_seq2seq(self)
    
    def forward(self, X, *args):
        embs = self.embedding(X)
        output, h_n = self.rnn(embs)
        
        return output, h_n
        
    
