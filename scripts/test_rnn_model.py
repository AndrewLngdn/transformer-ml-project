from char_lstm import CharLSTM, CharLSTMTie
from jeeves_datasets import get_jeeves_datasets
from rnn_model import NextCharRNN
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import torch

torch.manual_seed(42)


# model_config = {
#     "model_type": "RNN",
#     "embedding_dim": 128,
#     "hidden_size": 64,
#     "lr": 1e-3,
#     "grad_clip": True,
#     "batch_size": 16,
#     "seq_len": 64
# }

datasets = get_jeeves_datasets(T=64) 
train_dataset = datasets.train_dataset
val_dataset = datasets.val_dataset
test_dataset = datasets.test_dataset

vocab_len = datasets.vocab_size
batch_size = 16
print(f"Length of train dataset: {len(train_dataset)=}")
print(f"Sequence length T: {datasets.T}")

train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# MODEL_TYPE = "LSTM"
# if MODEL_TYPE == "LSTM":
#     model = CharLSTM(vocab_size=vocab_len)
# else:
#     model = NextCharRNN(vocab_size=vocab_len)
# model = NextCharRNN(vocab_size=vocab_len, hidden_size=64, embedding_dim=64)

model = CharLSTMTie(vocab_size=vocab_len)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


NUM_EPOCHS = 20
step = 0
ema_smoothing = 0.5

model.to(device)


for epoch in tqdm(range(NUM_EPOCHS)):
    
    total_epoch_train_loss = 0
    train_token_count = 0
    
    model.train()
    
    for x_train, y_train in train_dl:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(x_train)
        
        y_pred = torch.reshape(y_pred, (-1, vocab_len))
        y_actual = torch.reshape(y_train, (-1,))
        
        loss = loss_fn(y_pred, y_actual)
        loss.backward()
        
        loss_scalar = loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        train_token_count += y_actual.shape[0]
        total_epoch_train_loss += loss_scalar * y_actual.shape[0]
        
        if step == 0:
            print(f"First batch loss: {loss_scalar:.2f}. {-np.log(1/vocab_len)=:.2f}")
            
        step += 1

        
    avg_train_loss_per_char = total_epoch_train_loss / train_token_count
    
    
    model.eval()
    
    total_epoch_valid_loss = 0
    valid_token_count = 0
    best_val_loss = float("inf")
    
    with torch.no_grad():
        for x_valid, y_valid in valid_dl:
            x_valid = x_valid.to(device)
            y_valid = y_valid.to(device)
            
            y_pred = model(x_valid)
            
            y_pred = torch.reshape(y_pred, (-1, vocab_len))
            y_actual = torch.reshape(y_valid, (-1,))
            
            loss = loss_fn(y_pred, y_actual)
            loss_scalar = loss.item()
            
            valid_token_count += y_actual.shape[0]
            total_epoch_valid_loss += loss_scalar * y_actual.shape[0]

        avg_val_loss_per_char = total_epoch_valid_loss / valid_token_count

    print(f"epoch {epoch} | train: {avg_train_loss_per_char:.2f} | val: {(total_epoch_valid_loss / valid_token_count):.2f} | val bpc: {avg_val_loss_per_char / np.log(2)}")
        
        
        
    