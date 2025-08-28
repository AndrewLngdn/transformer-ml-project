
from char_data import CharDataset, build_char_vocab, decode, encode, split_ids
from data_importer import read_txt

my_man_jeeves = read_txt("data/my_man_jeeves.txt")
right_ho_jeeves = read_txt("data/right_ho_jeeves.txt")
the_inimitable_jeeves = read_txt("data/the_inimitable_jeeves.txt")

train_and_val_str = my_man_jeeves + "\n\n" + right_ho_jeeves 
test_str = the_inimitable_jeeves

itos, stoi = build_char_vocab(train_and_val_str + test_str)

train_val_tokens = encode(train_and_val_str, stoi)
test_tokens = encode(test_str, stoi)
train_tokens, val_tokens = split_ids(train_val_tokens, frac=0.8)

train_dataset = CharDataset(train_tokens, T=16)
val_dataset = CharDataset(val_tokens, T=16)
test_dataset = CharDataset(test_tokens, T=16)

