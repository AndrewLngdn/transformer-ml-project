from hello_data import make_dataloaders
from hello_model import HelloModel
import torch
from torch.optim import Adam

VOCAB_SIZE = 128
train_dl, test_dl = make_dataloaders(VOCAB_SIZE)

EMBEDDING_DIM = 64

model = HelloModel(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
model.eval()


loss_fn = torch.nn.CrossEntropyLoss()

# loss = loss_fn(output_batch, y_train)

total_loss = 0

for x_train, y_train in train_dl:
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    
    
    total_loss += loss.item() * x_train.shape[0]

pre_train_loss = total_loss / len(train_dl.dataset)
print(f"{pre_train_loss=}")

model.train()
optimizer = Adam(model.parameters(), lr=1e-2)
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    total_batch_loss = 0
    model.train()
    
    for x_train, y_train in train_dl:
        optimizer.zero_grad()
        
        y_pred = model(x_train)
        
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        
        optimizer.step()
        
        total_batch_loss += loss.item() * x_train.shape[0]

    avg_train_loss_per_item = total_batch_loss / len(train_dl.dataset)

    print(f"{avg_train_loss_per_item=}")
    
    model.eval()
    total_test_loss = 0.0
    total_test_count = 0
    for x_test, y_test in test_dl:
        with torch.no_grad():
            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)
            total_test_loss += loss.item() * x_test.size(0)
            total_test_count += x_test.size(0)
            
    avg_test_loss_per_item = total_test_loss / total_test_count
    print(f"{avg_test_loss_per_item=}")
    print()

