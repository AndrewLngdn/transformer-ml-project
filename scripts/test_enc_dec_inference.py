import evaluate
from data_importer import read_txt
from decoder import AttentionSeq2SeqDecoder, Seq2SeqDecoder
from encoder import Seq2SeqEncoder
from encoder_decoder import EncoderDecoder
from translation_data import EOS, FraEngDatasets, BOS, decode_batch_py, decode_one
import torch 
from torch.utils.data import DataLoader
from pprint import pprint

path="data/fra_clean.txt"
text = read_txt(path)

import time

def sync_device():
    """Synchronize the device so timings are accurate."""
    import torch
    torch.mps.synchronize()

class Timer:
    def __enter__(self):
        sync_device()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        sync_device()
        self.dt = time.perf_counter() - self.t0


if torch.backends.mps.is_available() and False:
    print("here")
    device = torch.device("mps")
else:
    device = torch.device("cpu")


num_epochs = 30
batch_size = 8

seq_len = 32
emb_dim = 256
hidden_dim = 256
num_layers = 2
dropout = 0.2

fe_datasets = FraEngDatasets(text, seq_len=seq_len)

input_vocab_size = len(fe_datasets.src_itos)
output_vocab_size = len(fe_datasets.tgt_itos)


train_ds = fe_datasets.train_dataset
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_ds = fe_datasets.val_dataset
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)


encoder = Seq2SeqEncoder(vocab_size=input_vocab_size, embed_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
decoder = AttentionSeq2SeqDecoder(vocab_size=output_vocab_size, emb_size=emb_dim, attn_hidden_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)

model = EncoderDecoder(encoder, decoder)

if False:
    model.load_state_dict(torch.load("checkpoints/enc_dec.model", weights_only=True))
    
model.eval()
model = model.to(device)


bos = fe_datasets.tgt_stoi[BOS]
eos = fe_datasets.tgt_stoi[EOS]

count = 0

with Timer() as t:
    with torch.no_grad():

        for src, tgt in valid_dl:
            src = src.to(device)
            tgt = tgt.to(device)
            
            state = model.encoder(src)
            
            batch_size = src.shape[0]
            
            count += 1
            if count == 10:
                break
            
            # src = src.unsqueeze(dim=0)
            
            dec_input = tgt[:, :1]
            seqs = torch.zeros(batch_size, seq_len, 1, dtype=torch.int64)
            
            for i in range(seq_len):
                out, state = model.decoder(dec_input, state)
                dec_input = torch.argmax(out, dim=2)
                seqs[:, i] = dec_input
                
                print(".", end="")

            break
print()



def pprint_tokens(token_lists):
    for token_list in token_lists:
        clean_list = []
        for token in token_list:
            if token == EOS:
                break
            if token == BOS:
                continue
            clean_list.append(token)
            
        print(" ".join(clean_list), end=" ")
    print()
sacrebleu = evaluate.load("sacrebleu")


def tokens_to_str(token_list):
    clean_list = []

    for token in token_list:
        if token == EOS:
            break
        if token == BOS:
            continue
        clean_list.append(token)
    return " ".join(clean_list) or " "
       
src_decoded = (decode_batch_py(src, fe_datasets.src_itos))
tgt_decoded = (decode_batch_py(tgt, fe_datasets.tgt_itos))
seqs_decoded = (decode_batch_py(seqs.squeeze(), fe_datasets.tgt_itos))

src_strs = [tokens_to_str(t) for t in src_decoded]
tgt_strs = [tokens_to_str(t) for t in tgt_decoded]
seqs_strs = [tokens_to_str(t) for t in seqs_decoded]
predictions = seqs_strs
references = [[s] for s in tgt_strs]

print(predictions)
print(references)
sacrebleu = evaluate.load("sacrebleu")
results = sacrebleu.compute(predictions=predictions, references=references)

print(results)
print(f"Elapsed: {t.dt:.4f} seconds")
breakpoint()
