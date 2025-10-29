from transformer import TransformerDecoderBlock, TransformerEncoderBlock, TransformerEncoder
import torch


B, L, D = 2, 5, 32

vocab_size = 100
num_heads = 4
num_blocks = 3
dropout = 0.2

enc_block = TransformerEncoderBlock(D, num_heads, 2 * D, 0.2)


input = torch.randn((B, L, D))
valid_lens = torch.randint(1, L, size=(B,))
range_ = torch.arange(L)[None, :]
mask = range_ < valid_lens[:, None]

input[~mask] = 0

out = enc_block(input, valid_lens)
print("TransformerEncoderBlock")
print(f"{out.shape=}")

print("TransformerEncoder:")
encoder = TransformerEncoder(
    vocab_size=vocab_size, 
    embed_dim=D, 
    num_heads=num_heads, 
    num_blocks=num_blocks, 
    dropout=dropout
)
enc_X = torch.arange(0, B * L, dtype=torch.long).reshape((B, L))
print(f"{encoder=}")

print(f"{encoder(enc_X, valid_lens).shape=}")

enc_out = encoder(enc_X, valid_lens)

print()

decoder_block0 = TransformerDecoderBlock(embed_dim=D, num_heads=num_heads, ffn_num_hiddens=2 * D, dropout=dropout, i=0)
decoder_block1 = TransformerDecoderBlock(embed_dim=D, num_heads=num_heads, ffn_num_hiddens=2 * D, dropout=dropout, i=1)
decoder_block2 = TransformerDecoderBlock(embed_dim=D, num_heads=num_heads, ffn_num_hiddens=2 * D, dropout=dropout, i=2)

state = [enc_out, valid_lens, [None] * 3]

decoder_block0.training = False
decoder_block1.training = False
decoder_block2.training = False

dec_seq = torch.randn((B, 1, D))

for i in range(L):
    dec_seq, state = decoder_block0(dec_seq, state)
    dec_seq, state = decoder_block1(dec_seq, state)
    dec_seq, state = decoder_block2(dec_seq, state)


print(f"{dec_seq.shape=}")
print()