from jeeves_datasets import stoi, itos, tokens, dataset
from rnn_model import NextCharRNN
from torch import nn
from torch.utils.data import DataLoader

import torch

vocab_len = len(stoi)
batch_size = 64
print(f"Length of dataset: {len(dataset)=}")

train_dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
# x_train, y_train = next(iter(train_dl))
model = NextCharRNN(vocab_size=vocab_len)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# y_pred = model(x_train)

model.train()

NUM_EPOCHS = 50
step = 0
ema_smoothing = 0.1
for epoch in range(NUM_EPOCHS):
    
    total_epoch_loss = 0
    data_point_count = 0
    
    for x_train, y_train in train_dl:
        optimizer.zero_grad()
        
        y_pred = model(x_train)
        
        y_pred = torch.reshape(y_pred, (-1, vocab_len))
        y_actual = torch.reshape(y_train, (-1,))
        
        loss = loss_fn(y_pred, y_actual)
        loss.backward()
        
        loss_scalar = loss.item()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_epoch_loss += loss_scalar * x_train.shape[0]
        data_point_count += x_train.shape[0]
        
        
        if step == 0:
            ema = loss_scalar
        else:
            ema = ema_smoothing * loss_scalar + (1 - ema_smoothing) * ema
            
        if step % 50 == 0:
            print(f"STEP: {step} {ema=:.3f}")
        step += 1
        
    avg_loss_per_item = total_epoch_loss / data_point_count
    
    print(f"EPOCH: {epoch} train: {avg_loss_per_item=:.3f}")
    
    
    model.eval()
    