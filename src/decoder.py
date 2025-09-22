from torch import nn
import torch

from attention import AdditiveAttention, MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X, *args):
        raise NotImplementedError


class AttentionSeq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.rnn = nn.GRU(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.dense = nn.LazyLinear(vocab_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads=4, dropout=dropout)

    def init_state(self, enc_all_outputs, *args):
        self.enc_all_outputs = enc_all_outputs
        return enc_all_outputs

    def attention_weights(self):
        return self.attention.attention_scores
    
    def forward(self, X, state, seq_lens=None):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(X)
        enc_output, hidden_state = state
        
        dec_output, h_n = self.rnn(embs, hidden_state)
        
        attended_states = self.attention(dec_output, enc_output, enc_output, seq_lens)
        
        dense_input = torch.cat((dec_output, attended_states), dim=-1)
        logits = self.dense(dense_input)
        
        return logits, [enc_output, h_n]

class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.rnn = nn.GRU(emb_size + hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(hidden_size, vocab_size)
        
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs
    
    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(X)
        enc_output, hidden_state = state
        
        seq_len = X.shape[1]
        context = enc_output[:, -1]
        context = context.unsqueeze(1).repeat(1, seq_len, 1)
        
        dec_input = torch.cat((embs, context), -1)
        
        dec_output, h_n = self.rnn(dec_input, hidden_state)
        output = self.dense(dec_output)
        
        return output, [enc_output, h_n]

        