from hello_data import ToyNextIntDataset

tds = ToyNextIntDataset(10)

tds_iterator = iter(tds)

for i in range(11):
    print(tds[i])
# while tds_iterator:
#     print(next(tds_iterator))