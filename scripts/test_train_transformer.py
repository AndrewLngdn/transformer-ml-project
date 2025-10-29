
from tqdm import tqdm
from data_importer import read_txt
from decoder import AttentionSeq2SeqDecoder, Seq2SeqDecoder
from encoder import Seq2SeqEncoder
import torch
from torch.utils.data.dataloader import DataLoader
import math

from encoder_decoder import EncoderDecoder, TransformerEncoderDecoder
from transformer import TransformerDecoder, TransformerEncoder
from translation_data import FraEngDatasets, PAD
from transformers import get_cosine_schedule_with_warmup


from torch.utils.data import Subset

from torch.utils.data import Subset



if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


path="data/fra_clean.txt"
text = read_txt(path)

num_epochs = 30
batch_size = 32

seq_len = 16
emb_dim = 256
hidden_dim = 256
ffn_num_hiddens = 64
num_heads = 4
num_blks = 2
dropout = 0.2
lr = 5e-5

print(f"{batch_size=}")
print(f"{seq_len=}")
print(f"{emb_dim=}")
print(f"{hidden_dim=}")
print(f"{ffn_num_hiddens=}")
print(f"{num_heads=}")
print(f"{num_blks=}")
print(f"{dropout=}")
print(f"{lr=}")

fe_datasets = FraEngDatasets(text, seq_len=seq_len)

input_vocab_size = len(fe_datasets.src_itos)
print(f"{input_vocab_size=}")
output_vocab_size = len(fe_datasets.tgt_itos)
print(f"{output_vocab_size=}")



train_ds = fe_datasets.train_dataset
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_ds = fe_datasets.val_dataset
# valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

import random
# indices = random.sample(range(len(train_ds)), 8000)
small_train = fe_datasets.train_dataset
# small_val = fe_datasets.val_dataset

# small_train = Subset(train_ds, indices)
small_val   = Subset(valid_ds, range(2000))



train_dl = DataLoader(small_train, batch_size=batch_size, shuffle=True)
valid_dl   = DataLoader(small_val, batch_size=batch_size)


encoder = TransformerEncoder(vocab_size=input_vocab_size, embed_dim=emb_dim, num_heads=num_heads, num_blocks=num_blks, dropout=dropout)
decoder = TransformerDecoder(vocab_size=output_vocab_size, embed_dim=emb_dim, num_heads=num_heads, num_blocks=num_blks, dropout=dropout)

input_PAD = fe_datasets.src_stoi[PAD]


model = TransformerEncoderDecoder(encoder, decoder)
model = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=fe_datasets.tgt_stoi[PAD], label_smoothing=0.1)
valid_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=fe_datasets.tgt_stoi[PAD])

total_steps = num_epochs * math.ceil(len(train_dl))
warmup_steps = int(0.1 * total_steps)  # 5% of training

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=0.1
)

train_losses = []

model.eval()
with torch.no_grad():
    src, tgt = next(iter(train_dl))
    src, tgt = src.to(device), tgt.to(device)
    seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)
    print(f"{src.shape=}")

    logits = model(src, tgt, seq_lens=seq_lens)                     # or (src, tgt[:, :-1]) if you handle shift outside
    L = torch.nn.functional.cross_entropy(
        logits.reshape(-1, output_vocab_size),
        tgt.reshape(-1),
        ignore_index=fe_datasets.tgt_stoi[PAD]
    )
    print("Init loss (should be ~ln(V)):", L.item())    


step = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_train_loss = 0
    import time

    for src, tgt in train_dl:
        src = src.to(device)
        tgt = tgt.to(device)

        dec_in = tgt[:, :-1]
        actual = tgt[:, 1:]
        
        seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)

        with torch.autocast(device_type="mps", dtype=torch.float16):
            output = model(src, dec_in, seq_lens) 
            actual = actual.reshape(-1)
            output = output.reshape(-1, output_vocab_size)
            loss = loss_fn(output, actual)
            
            if not torch.isfinite(loss):
                print("⚠️ Non-finite loss detected — checking activations and gradients")
                for name, p in model.named_parameters():
                    if not torch.isfinite(p).all():
                        print(f"Param {name} contains NaN/Inf!")
        
        optimizer.zero_grad()
        loss.backward()
        # Compute global grad norm (L2 norm across all parameters)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, error_if_nonfinite=False)
        train_losses.append(loss.item())
        

        # Log only when clipping *would* activate
        if total_norm > 10.0:
            print(f" | norm = {total_norm:.2f}", end="")

        optimizer.step()
        scheduler.step()
        
        step += 1
        
        softmax = torch.max(torch.softmax(output, dim=1), dim=1)
        
        epoch_train_loss += loss.item()
        if step % 3000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nStep {step} | Current LR: {current_lr:.6f}")
            

    print(f"Epoch {epoch} step {step} train loss: {loss.item()}")

    valid_batch_losses = []
    model.eval()
    
    best_val = float("inf")
    best_val_epoch = 0
    
    for src, tgt in valid_dl:
        with torch.no_grad():
            src = src.to(device)
            tgt = tgt.to(device)
            
            dec_in = tgt[:, :-1]
            actual = tgt[:, 1:]
            
            seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)
            
            output = model(src, dec_in, seq_lens)
            actual = actual.reshape(-1)
            output = output.reshape(-1, output_vocab_size)
            
            loss = valid_loss_fn(output, actual)
            
            if loss.item() < best_val:
                best_val = loss.item()
                best_val_epoch = epoch
            
            
            if epoch - best_val_epoch > 3:
                print(f"Early stopping!")
                raise Exception("Early stopping!")
            
            valid_batch_losses.append(loss.item())
    
    avg_valid_loss = sum(valid_batch_losses) / len(valid_batch_losses)
    
    if avg_valid_loss < best_val:
        
        torch.save(model.state_dict(), f"checkpoints/transformer.model")
    
    print(f"Epoch {epoch} valid avg batch loss: {avg_valid_loss}")
    
            
            
    
    



# despy - 