import pytest
import torch
from src.char_data import build_char_vocab, encode, decode, split_ids, CharDataset

# Test data string
s = "Jeeves (born Reginald Jeeves, nicknamed Reggie[1]) is a fictional character in a series of comedic short stories and novels by the English author P. G. Wodehouse. Jeeves is the highly competent valet of a wealthy and idle young Londoner named Bertie Wooster. First appearing in print in 1915, Jeeves continued to feature in Wodehouse's work until his last completed novel, Aunts Aren't Gentlemen, in 1974. "


class TestCharData:
    """Smoke tests for character-level data processing functions."""
    
    @pytest.mark.unit
    def test_vocab_encode_decode_roundtrip(self, sample_text):
        """Test basic vocabulary building and encode/decode roundtrip."""
        itos, stoi = build_char_vocab(sample_text)
        encoded = encode(sample_text, stoi)
        decoded = decode(encoded, itos)
        
        assert decoded == sample_text
        assert len(itos) == len(set(sample_text))
        
    @pytest.mark.unit  
    def test_split_ids_works(self, sample_text):
        """Test that split_ids produces train/val splits."""
        itos, stoi = build_char_vocab(sample_text)
        encoded = encode(sample_text, stoi)
        train, val = split_ids(encoded)
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(train) + len(val) == len(encoded)
        
    @pytest.mark.unit
    def test_char_dataset_works(self, sample_text):
        """Test that CharDataset can be created and produces samples."""
        itos, stoi = build_char_vocab(sample_text)
        encoded = encode(sample_text, stoi)
        dataset = CharDataset(encoded, T=5)
        
        x, y = dataset[0]
        assert len(x) == 5
        assert len(y) == 5
        
    @pytest.mark.unit
    def test_main_functionality(self):
        """Test the main functionality from __main__ block."""
        # Build vocabulary
        itos, stoi = build_char_vocab(s)
        assert len(itos) > 0
        assert len(stoi) > 0
        
        # Test encode/decode roundtrip
        code = encode(s, stoi)
        decoded = decode(code, itos)
        assert decoded == s
        
        # Test unknown character raises KeyError
        with pytest.raises(KeyError):
            encode(s + "€", stoi)
        
        # Test split
        train, val = split_ids(code)
        combined = torch.cat((train, val))
        assert torch.equal(combined, code)
        
        # Test dataset
        ds = CharDataset(tokens=train, T=16)
        x_train, y_train = ds[0]
        assert len(x_train) == 16
        assert len(y_train) == 16
        assert torch.equal(x_train[1:], y_train[:-1])