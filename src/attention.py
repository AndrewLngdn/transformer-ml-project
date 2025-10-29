import torch



# tensor([[[0.4448, 0.5552, 0.0000, 0.0000],
#          [0.4032, 0.5968, 0.0000, 0.0000]],

#         [[0.2795, 0.2805, 0.4400, 0.0000],
#          [0.2798, 0.3092, 0.4110, 0.0000]]])

import torch

def masked_softmax(batched_X, lengths=None):
    """
    batched_X: (B, H, T) logits
    lengths:   None, (B,), or (B,H) specifying how many positions are valid
    Returns:   (B, H, T) probabilities; rows that are fully masked are all zeros.
    """
    if lengths is None:
        return torch.softmax(batched_X, dim=-1)

    if lengths.ndim == 1:          # (B,)
        lengths = lengths[:, None] # -> (B,1)
    elif lengths.ndim == 2:        # (B,H)
        pass
    else:
        raise ValueError("lengths must be shape (B,) or (B,H)")

    B, H, T = batched_X.shape
    device = batched_X.device

    # Build mask via broadcasting: positions (T,) vs lengths (...,1)
    positions = torch.arange(T, device=device)
    mask = positions < lengths[..., None]   # -> (B,1,T) or (B,H,T)

    # Mask logits with -inf (avoid modifying input)
    neg_inf = torch.finfo(batched_X.dtype).min
    masked_logits = batched_X.masked_fill(~mask, neg_inf)

    # Softmax over last dim
    probs = torch.softmax(masked_logits, dim=-1)

    # Handle fully-masked rows: if no True in a row, make the whole row zeros
    fully_masked = ~mask.any(dim=-1, keepdim=True)  # (B,H,1)
    probs = torch.where(fully_masked, torch.zeros_like(probs), probs)

    return probs




class DotProductAttention(torch.nn.Module):
    
    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.attention_weights = None
    
    def forward(self, queries, keys, values, valid_lens=None):
        # Shape of queries: (batch_size, no. of queries, d)
        # Shape of keys: (batch_size, no. of key-value pairs, d)
        # Shape of values: (batch_size, no. of key-value pairs, value dimension)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        scale = queries.shape[-1] ** 0.5
        
        QK_t = torch.bmm(queries, keys.transpose(1, 2)) 
        scores = QK_t / scale 
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        output = torch.bmm(self.dropout(self.attention_weights), values)
        
        return output
    
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise Exception("hidden dim must be divisible by num heads")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.dropout = dropout
        self.bias = bias
        
        self.head_dim = hidden_dim // num_heads
        
    
        self.W_k = torch.nn.Linear(hidden_dim, self.num_heads * self.head_dim, bias=bias)
        self.W_q = torch.nn.Linear(hidden_dim, self.num_heads * self.head_dim, bias=bias)
        self.W_v = torch.nn.Linear(hidden_dim, self.num_heads * self.head_dim, bias=bias)
        
        self.W_out = torch.nn.Linear(self.num_heads * self.head_dim, hidden_dim, bias=bias)
        
        self.attention = DotProductAttention(dropout)
        
    def forward(self, queries, keys, values, valid_lens=None):
        B, Lq, _ = queries.shape
        B, Lk, _ = keys.shape
        q_proj = self.W_q(queries)
        k_proj = self.W_k(keys)
        v_proj = self.W_v(values)
        
        # shape: (B, L, h * d_head) -> (B, L, h, d_head) -> (B, h, L, d_head)
        # is this wrong?   
        q_proj = torch.reshape(q_proj, (B, Lq, self.num_heads, self.head_dim)).transpose(1, 2).reshape(self.num_heads * B, Lq, self.head_dim)
        k_proj = torch.reshape(k_proj, (B, Lk, self.num_heads, self.head_dim)).transpose(1, 2).reshape(self.num_heads * B, Lk, self.head_dim)
        v_proj = torch.reshape(v_proj, (B, Lk, self.num_heads, self.head_dim)).transpose(1, 2).reshape(self.num_heads * B, Lk, self.head_dim)
        
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        
        attn_out = self.attention(q_proj, k_proj, v_proj, valid_lens)
        # if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
        #     print("⚠️ NaN or Inf in attention scores!")
        # else:
        #     print(f"[attn] mean={attn_out.mean():.3f}, std={attn_out.std():.3f}, max={attn_out.max():.3f}")

        attn_out = attn_out.reshape(B, self.num_heads, Lq, self.head_dim).transpose(1,2).reshape(B, Lq, self.num_heads * self.head_dim)

        out = self.W_out(attn_out)
        return out
    
    
class AdditiveAttention(torch.nn.Module):
    def __init__(self, num_hiddens, dropout, **kwargs):
        super().__init__()
        self.w_v = torch.nn.LazyLinear(1, bias=False)
        self.W_q = torch.nn.LazyLinear(num_hiddens, bias=False)
        self.W_k = torch.nn.LazyLinear(num_hiddens, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, valid_lens=None):
        q_proj = self.W_q(queries)
        k_proj = self.W_k(keys)
        
        hidden_sum = q_proj.unsqueeze(2) + k_proj.unsqueeze(1)
        features = torch.tanh(hidden_sum)
        scores = self.w_v(features).squeeze()
        
        self.attention_scores = masked_softmax(scores, valid_lens)
        output = torch.bmm(self.dropout(self.attention_scores), values)
        return output
        
        
        
if __name__ == "__main__":
    queries = torch.normal(0, 1, (2, 2, 20))

    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])

    # attention = DotProductAttention(dropout=0.5)
    # attention.eval()
    
    attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))
    
    
# queries =
# [[ 0,  1],
#  [ 4,  5]

# keys =
# [[ 0,  1],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]