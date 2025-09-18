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
        scale = queries.shape[-1] / 0.5
        
        QK_t = torch.bmm(queries, keys.transpose(1, 2)) 
        scores = QK_t / scale 
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        output = torch.bmm(self.dropout(self.attention_weights), values)
        
        return output
    
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