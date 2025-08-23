from hello_data import ToyNextIntDataset, make_dataloaders
from torch.utils.data import DataLoader

tds = ToyNextIntDataset(10)

tds_iterator = iter(tds)

try: 
    for i in range(11):
        print(tds[i])
except IndexError as e:
    print("DS raised index error as expected:")
    print(e)
    print()

tds_dl = DataLoader(tds, batch_size=4)

x_tensor, y_tensor = next(iter(tds_dl))

print(f"{x_tensor=}")
print(f"{x_tensor.shape=}")
print(f"{y_tensor=}")
print(f"{y_tensor.shape=}")

train_dl, test_dl = make_dataloaders(N=64)

print("--- DataLoaders ---")
for train_x, train_y in train_dl:
    print(f"{train_x=}")
    print(f"{train_y=}")
    
for test_x, test_y in test_dl:
    print(f"{test_x=}")
    print(f"{test_y=}")