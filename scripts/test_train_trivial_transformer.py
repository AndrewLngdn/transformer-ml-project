
from tqdm import tqdm
from data_importer import read_txt
import torch
from torch.utils.data.dataloader import DataLoader

from encoder_decoder import EncoderDecoder, TransformerEncoderDecoder
from transformer import TransformerDecoder, TransformerEncoder
from translation_data import FraEngDatasets, PAD
from trivial_dataset import TrivialDataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

num_epochs = 300
batch_size = 2

seq_len = 4
emb_dim = 256
hidden_dim = 256
ffn_num_hiddens = 64
num_heads = 4
num_blks = 1
dropout = 0.0


ds_vocab_size = 5
seq_len = 8
dataset_len = 200

trivial_ds = TrivialDataset(vocab_size=ds_vocab_size, seq_len=seq_len, dataset_len=dataset_len, seq_len_min=1)
trivial_dl = DataLoader(dataset=trivial_ds, shuffle=True, batch_size=batch_size)


encoder = TransformerEncoder(vocab_size=ds_vocab_size, embed_dim=emb_dim, num_heads=num_heads, num_blocks=num_blks, dropout=dropout)
decoder = TransformerDecoder(vocab_size=ds_vocab_size, embed_dim=emb_dim, num_heads=num_heads, num_blocks=num_blks, dropout=dropout)

input_PAD = trivial_ds.PAD


model = TransformerEncoderDecoder(encoder, decoder)
model = model.to(device)


model.eval()
with torch.no_grad():
    src, tgt = next(iter(trivial_dl))
    src, tgt = src.to(device), tgt.to(device)
    seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)
    print(f"{src.shape=}")

    logits = model(src, tgt, seq_lens=seq_lens)                     # or (src, tgt[:, :-1]) if you handle shift outside
    L = torch.nn.functional.cross_entropy(
        logits.reshape(-1, ds_vocab_size),
        tgt.reshape(-1),
        ignore_index=input_PAD
    )
    print("Init loss (should be ~ln(V)):", L.item())    

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=input_PAD)

step = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_train_loss = 0

    for src, tgt in trivial_dl:

        src = src.to(device)
        tgt = tgt.to(device)

        dec_in = tgt[:, :-1]
        actual = tgt[:, 1:]
        
        seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)

        output = model(src, dec_in, seq_lens) 
        
        actual = actual.reshape(-1)
        
        output = output.reshape(-1, ds_vocab_size)
        

        loss = loss_fn(output, actual)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        
        softmaxes = torch.max(torch.softmax(output, dim=1), dim=1)
        
        # print(f"{}")
        # print(f"{actual=}")
        
        epoch_train_loss += loss.item()
        
    print(f"Epoch {epoch} step {step} train loss: {loss.item()}")

    # valid_batch_losses = []
    # model.eval()
    
    # best_val = float("inf")
    
    # for src, tgt in valid_dl:
    #     with torch.no_grad():
    #         src = src.to(device)
    #         tgt = tgt.to(device)
            
    #         dec_in = tgt[:, :-1]
    #         actual = tgt[:, 1:]
            
    #         seq_lens = torch.where(src != input_PAD, 1, 0).sum(dim=1)
            
    #         output = model(src, dec_in, seq_lens)
    #         actual = actual.reshape(-1)
    #         output = output.reshape(-1, output_vocab_size)
            
    #         loss = loss_fn(output, actual)
            
    #         valid_batch_losses.append(loss.item())
    
    # avg_valid_loss = sum(valid_batch_losses) / len(valid_batch_losses)
    
    # if avg_valid_loss < best_val:
        
    #     torch.save(model.state_dict(), f"checkpoints/transformer.model")
    
    # print(f"Epoch {epoch} valid avg batch loss: {avg_valid_loss}")
    
            
            
    
    



# despy - 