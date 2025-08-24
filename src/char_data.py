import torch
from typing import Tuple, Dict, List

s = "Jeeves (born Reginald Jeeves, nicknamed Reggie[1]) is a fictional character in a series of comedic short stories and novels by the English author P. G. Wodehouse. Jeeves is the highly competent valet of a wealthy and idle young Londoner named Bertie Wooster. First appearing in print in 1915, Jeeves continued to feature in Wodehouse's work until his last completed novel, Aunts Aren't Gentlemen, in 1974. "


def build_char_vocab(s: str) -> Tuple[List[str], Dict[str, int]]:
    itos = sorted(set(s))
    
    stoi = {}
    
    for i, char in enumerate(itos):
        stoi[char] = i
        
    return itos, stoi



def encode(s: str, stoi) -> torch.Tensor:
    ints = []
    for c in s:
        try:
            ints.append(stoi[c])
        except KeyError as e:
            print(f"{c} isn't in vocabulary")
            raise e
            
    return torch.tensor(ints, dtype=torch.int64)


def decode(t: torch.Tensor, itos) -> str:
    ints = t.tolist()
    chars = []
    
    for int in ints:
        chars.append(itos[int])
    
    return "".join(chars)


def split_ids(chars_tensor: torch.Tensor, frac=0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    train_len = int(len(chars_tensor) * frac)
    
    train_tensor = chars_tensor[:train_len]
    val_tensor = chars_tensor[train_len:]
    
    return train_tensor, val_tensor


if __name__ == "__main__":
    
    
    itos, stoi = build_char_vocab(s)

    print(f"{itos=}")co
    print(f"{stoi=}")


    code = encode(s, stoi)
    decoded = decode(code, itos)
    print(f"{(decoded == s)=}")

    try:
        encode(s + "€", stoi)
        
    except KeyError as e:
        print(e)
        


    train, val = split_ids(code)

    print(f"{train=}")
    print(f"{val=}")

    combined = torch.cat((train, val))
    assert torch.equal(combined, code)


