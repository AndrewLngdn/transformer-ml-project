from torch.utils.data import Dataset
from char_data import split_ids
from data_importer import read_txt
import torch
from pprint import pprint

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


STOI_BASE = {
    PAD: 0,
    BOS: 1,
    EOS: 2,
    UNK: 3
}

def _preprocess(text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)

def _tokenize(text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append(["<bos>"] + [t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt


def _pad_or_trunc(src, tgt, seq_len, padding_token="<pad>"):
    assert len(src) == len(tgt), "src and target are diff lengths"
    
    for i in range(len(src)):
        src_tokens, tgt_tokens = src[i], tgt[i]
        
        if len(src_tokens) > seq_len:
            src_tokens[seq_len - 1] = "<eos>"
            src[i] = src_tokens[:seq_len]
        else:
            while len(src_tokens) < seq_len:
                src_tokens.append(padding_token)

        if len(tgt_tokens) > seq_len:
            tgt_tokens[seq_len - 1] = "<eos>"
            tgt[i] = tgt_tokens[:seq_len]
        else:
            while len(tgt_tokens) < seq_len:
                tgt_tokens.append(padding_token)
                
    return src, tgt
        
def _mask_low_count_tokens(src, tgt, min_count=3, masking_token="<unk>"):
    token_counts = {}
    
    for side in (src, tgt):
        for row in side:
            for token in row:
                if token not in token_counts:
                    token_counts[token] = 1
                else:
                    token_counts[token] += 1
    
    for side in (src, tgt):
        for row in side:
            for i in range(len(row)):
                if token_counts[row[i]] < min_count:
                    row[i] = masking_token
    
    return src, tgt

def _build_vocabs(tokenized_sentences):
    stoi = STOI_BASE.copy()
    next_i = len(stoi)
    
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in stoi:
                stoi[token] = next_i
                next_i += 1
            
    itos = {}
    for k, v in stoi.items():
        itos[v] = k
        
    return stoi, itos

def encode(tokenized_sentences, stoi):
    encoded_seqs = []
    
    for sentence in tokenized_sentences:
        seq = []
        
        for token in sentence:
            seq.append(stoi[token])
        
        encoded_seqs.append(seq)
    
    return torch.tensor(encoded_seqs, dtype=torch.int64)

def decode(encoded_sentences, itos):
    encoded_sentences = encoded_sentences.tolist()
    decoded_seqs = []
    
    for sentence in encoded_sentences:
        seq = []
        for i in sentence:
            seq.append(itos[i])
        
        decoded_seqs.append(seq)
        
    return decoded_seqs

def decode_one(seq, itos):
    return [itos[i.item()] for i in seq]
    
    
def decode_batch_py(ids: torch.Tensor, itos: list[str]) -> list[list[str]]:
    # ids: (B, T)
    return [[itos[i] for i in row] for row in ids.detach().cpu().tolist()]
    
class FraEngDatasets:
    def __init__(self, text, seq_len=32, min_count=3):
        text = _preprocess(text)
        src, tgt = _tokenize(text)
        src, tgt = _pad_or_trunc(src, tgt, seq_len)
        src, tgt = _mask_low_count_tokens(src, tgt, min_count=3)
        
        
        src_stoi, src_itos = _build_vocabs(src)
        tgt_stoi, tgt_itos = _build_vocabs(tgt)

        encoded_src = encode(src, src_stoi)
        encoded_tgt = encode(tgt, tgt_stoi)
        
        train_src, val_src = split_ids(encoded_src)
        train_tgt, val_tgt = split_ids(encoded_tgt)
        
        self.train_src = train_src
        self.val_src = val_src
        
        self.train_tgt = train_tgt
        self.val_tgt = val_tgt
        
        self.src_stoi = src_stoi
        self.src_itos = src_itos
        self.tgt_stoi = tgt_stoi
        self.tgt_itos = tgt_itos
        
        self.train_dataset = FraEngDataset(self.train_src, self.train_tgt)
        self.val_dataset = FraEngDataset(self.val_src, self.val_tgt)
        

class FraEngDataset(Dataset):
    def __init__(self, src_seqs, tgt_seqs):
        assert len(src_seqs[0]) == len(tgt_seqs[0]), "sequences must be the same len"
        super().__init__()
        
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs
    
    def __len__(self):
        return len(self.src_seqs)
    
    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError()
        
        return self.src_seqs[i], self.tgt_seqs[i]

if __name__ == "__main__":
    path="data/fra_clean.txt"
    text = read_txt(path)
    
    fe_datasets = FraEngDatasets(text)
    
    for i, (src, tgt) in enumerate(fe_datasets.train_dataset):
        print("-" * 25)
        print(src)
        print(decode_one(src, fe_datasets.src_itos))
        print(tgt)
        print(decode_one(tgt, fe_datasets.tgt_itos))
        
        if i == 10:
            break
        
        
        
        
        
        