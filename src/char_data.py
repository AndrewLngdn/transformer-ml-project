import torch
from typing import Tuple, Dict, List

s = "Jeeves (born Reginald Jeeves, nicknamed Reggie[1]) is a fictional character in a series of comedic short stories and novels by the English author P. G. Wodehouse. Jeeves is the highly competent valet of a wealthy and idle young Londoner named Bertie Wooster. First appearing in print in 1915, Jeeves continued to feature in Wodehouse's work until his last completed novel, Aunts Aren't Gentlemen, in 1974. "


def build_char_vocab(s: str) -> Tuple[List[str], Dict[str, int]]:
    itos = sorted(set(s))
    
    stoi = {}
    
    for i, char in enumerate(itos):
        stoi[char] = i
        
    return itos, stoi


itos, stoi = build_char_vocab(s)

print(f"{itos=}")
print(f"{stoi=}")


def encode(s: str, stoi) -> torch.Tensor:
    ints = []
    for c in s:
        ints.append(stoi[c])
    return torch.tensor(ints, dtype=torch.int64)


def decode(t: torch.Tensor, itos) -> str:
    ints = list(t)
    chars = []
    
    for int in ints:
        try:
            chars.append(itos[int])
        except KeyError as e:
            print(f"{int} isn't in vocabulary")
    
    return "".join(chars)


code = encode(s, stoi)
decoded = decode(code, itos)
print(f"{(decoded == s)=}")
