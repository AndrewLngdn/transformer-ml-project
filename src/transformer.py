from torch import nn

from attention import MultiHeadAttention
from positional_encoding import LazySinCosPositionalEncoding
import torch

class PositionWiseFF(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape=norm_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_num_hiddens,  dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim=embed_dim, num_heads=num_heads, dropout=dropout, use_bias=use_bias)
        self.add_norm1 = AddNorm(embed_dim, dropout=dropout)
        self.ffn = PositionWiseFF(ffn_num_hiddens=ffn_num_hiddens, ffn_num_outputs=embed_dim)
        self.add_norm2 = AddNorm(embed_dim, dropout=dropout)
    
    
    def forward(self, X, valid_lens):
        # shape: B, L, D
        attn_out = self.attention(X, X, X, valid_lens)
        add_norm_1_out = self.add_norm1(attn_out, X)
        ffn_out = self.ffn(add_norm_1_out)
        out = self.add_norm2(ffn_out, add_norm_1_out)
        
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, dropout, use_bias=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.use_bias = use_bias
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.positional_encoding = LazySinCosPositionalEncoding(dropout=dropout, embed_dim=embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(TransformerEncoderBlock(
                embed_dim=embed_dim, num_heads=num_heads, ffn_num_hiddens=2 * embed_dim, dropout=dropout, use_bias=use_bias
            ))
        
        
        
    def forward(self, X, valid_lens):
        # Shape: B, seq_len
        embs = self.embeddings(X) * (self.embed_dim ** 0.5)
        out = self.positional_encoding(embs)
        
        for block in self.blocks:
            out = block(out, valid_lens)
        
        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_num_hiddens, dropout, i, use_bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.i = i
        
        self.dropout = nn.Dropout(dropout)
        
        self.masked_attention = MultiHeadAttention(hidden_dim=embed_dim, num_heads=num_heads, dropout=dropout, use_bias=use_bias)
        self.add_norm1 = AddNorm(embed_dim, dropout=dropout)
        
        self.enc_dec_attention = MultiHeadAttention(hidden_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.add_norm2 = AddNorm(embed_dim, dropout)
        
        self.ffn = PositionWiseFF(ffn_num_hiddens=ffn_num_hiddens, ffn_num_outputs=embed_dim)
        self.add_norm3 = AddNorm(embed_dim, dropout)
        

    def forward(self, X, state): # what should our valid lens be? 
        enc_output, enc_valid_lens, prev_dec_inputs = state
        
        # when training: X is whole sequence, need mask
        # when decoding: X is one token, need cache
        
        if prev_dec_inputs[self.i] is not None:
            keys_values = torch.cat((prev_dec_inputs[self.i], X), dim=1)
        else:
            keys_values = X
        
        prev_dec_inputs[self.i] = keys_values
        
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        masked_attn_out = self.masked_attention(X, keys_values, keys_values, dec_valid_lens)
        add_norm1_out = self.add_norm1(masked_attn_out, X)
        
        
        enc_dec_attn_out = self.enc_dec_attention(X, enc_output, enc_output, enc_valid_lens)
        
        add_norm2_out = self.add_norm2(enc_dec_attn_out, add_norm1_out)
        
        ffn_out = self.ffn(add_norm2_out)
        
        add_norm3_out = self.add_norm3(add_norm2_out, ffn_out)
        
        return add_norm3_out, state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, dropout, use_bias=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.num_blocks = num_blocks
        
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.positional_encoding = LazySinCosPositionalEncoding(dropout=dropout, embed_dim=embed_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(TransformerDecoderBlock(
                embed_dim=embed_dim, num_heads=num_heads, ffn_num_hiddens=2 * embed_dim, dropout=dropout, i=i, use_bias=use_bias
            ))
        
        self.ffn_out = nn.Linear(in_features=embed_dim, out_features=vocab_size)
            

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blocks]

    def forward(self, X, state):
        X = self.positional_encoding(self.embeddings(X) * (self.embed_dim ** 0.5))
        self._attention_weights = [[None] * len(self.blocks) for _ in range (2)]
        for i, blk in enumerate(self.blocks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.masked_attention.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.enc_dec_attention.attention.attention_weights
        return self.ffn_out(X), state

    @property
    def attention_weights(self):
        return self._attention_weights