
from char_data import CharDataset, build_char_vocab, decode, encode, split_ids
from data_importer import read_txt


class JeevesDatasets:
    """Manages Jeeves datasets with configurable sequence length."""
    
    def __init__(self, T=64, train_frac=0.8):
        """Initialize with sequence length T and train fraction."""
        self.T = T
        self.train_frac = train_frac
        
        # Load data
        my_man_jeeves = read_txt("data/my_man_jeeves.txt")
        right_ho_jeeves = read_txt("data/right_ho_jeeves.txt") 
        the_inimitable_jeeves = read_txt("data/the_inimitable_jeeves.txt")
        
        train_and_val_str = my_man_jeeves + "\n\n" + right_ho_jeeves
        test_str = the_inimitable_jeeves
        
        # Build vocabulary and encode
        self.itos, self.stoi = build_char_vocab(train_and_val_str + test_str)
        
        train_val_tokens = encode(train_and_val_str, self.stoi)
        test_tokens = encode(test_str, self.stoi)
        train_tokens, val_tokens = split_ids(train_val_tokens, frac=train_frac)
        
        # Create datasets
        self.train_dataset = CharDataset(train_tokens, T=T)
        self.val_dataset = CharDataset(val_tokens, T=T)
        self.test_dataset = CharDataset(test_tokens, T=T)
        
    @property
    def vocab_size(self):
        """Get vocabulary size."""
        return len(self.itos)


def get_jeeves_datasets(T=64, train_frac=0.8):
    """Get Jeeves datasets with configurable parameters."""
    return JeevesDatasets(T=T, train_frac=train_frac)


# For backward compatibility, create default datasets
try:
    _default = JeevesDatasets(T=64)
    train_dataset = _default.train_dataset
    val_dataset = _default.val_dataset  
    test_dataset = _default.test_dataset
    itos = _default.itos
    stoi = _default.stoi
except FileNotFoundError:
    # Set to None if files don't exist
    train_dataset = val_dataset = test_dataset = None
    itos = stoi = None

