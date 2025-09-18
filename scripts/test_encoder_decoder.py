
from tqdm import tqdm
from data_importer import read_txt
from decoder import AttentionSeq2SeqDecoder, Seq2SeqDecoder
from encoder import Seq2SeqEncoder
import torch
from torch.utils.data.dataloader import DataLoader

from encoder_decoder import EncoderDecoder
from translation_data import FraEngDatasets, PAD

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


path="data/fra_clean.txt"
text = read_txt(path)


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


encoder = Seq2SeqEncoder(vocab_size=input_vocab_size, embed_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout
                         )

decoder = AttentionSeq2SeqDecoder(vocab_size=output_vocab_size, emb_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)

input_PAD = fe_datasets.src_stoi[PAD]


model = EncoderDecoder(encoder, decoder)
model = model.to(device)


with torch.no_grad():
    src, tgt = next(iter(train_dl))
    src, tgt = src.to(device), tgt.to(device)
    seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)
    
    logits = model(src, tgt, seq_lens=seq_lens)                     # or (src, tgt[:, :-1]) if you handle shift outside
    L = torch.nn.functional.cross_entropy(
        logits.reshape(-1, output_vocab_size),
        tgt.reshape(-1),
        ignore_index=fe_datasets.tgt_stoi[PAD]
    )
    print("Init loss (should be ~ln(V)):", L.item()) 

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=fe_datasets.tgt_stoi[PAD])

step = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_train_loss = 0
    
    for src, tgt in train_dl:
        src = src.to(device)
        tgt = tgt.to(device)
        
        dec_in = tgt[:, :-1]
        actual = tgt[:, 1:]
        
        seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)

        output = model(src, dec_in, seq_lens)
        
        actual = actual.reshape(-1)
        
        output = output.reshape(-1, output_vocab_size)
        
        
        loss = loss_fn(output, actual)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        epoch_train_loss += loss.item()
        
        if step % 500 == 0:
            print(f"Epoch {epoch} step {step} train loss: {loss.item()}")

    valid_batch_losses = []
    model.eval()
    
    best_val = float("inf")
    
    for src, tgt in valid_dl:
        with torch.no_grad():
            src = src.to(device)
            tgt = tgt.to(device)
            
            dec_in = tgt[:, :-1]
            actual = tgt[:, 1:]
        
            output = model(src, dec_in)
            actual = actual.reshape(-1)
            output = output.reshape(-1, output_vocab_size)
            
            loss = loss_fn(output, actual)
            
            valid_batch_losses.append(loss.item())
    
    avg_valid_loss = sum(valid_batch_losses) / len(valid_batch_losses)
    
    if avg_valid_loss < best_val:
        
        torch.save(model.state_dict(), f"checkpoints/enc_dec_attn.model")
    
    print(f"Epoch {epoch} valid avg batch loss: {avg_valid_loss}")
    
            
            
    
    

