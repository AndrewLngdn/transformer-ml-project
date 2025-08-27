import pytest
import torch
from src.jeeves_datasets import itos, stoi, tokens, dataset, s


class TestJeevesDatasets:
    """Smoke tests for Jeeves dataset functionality."""
    
    @pytest.mark.unit
    def test_jeeves_text_exists(self):
        """Test that Jeeves text data is available."""
        assert len(s) > 0
        assert "Jeeves" in s
        assert "Cannes" in s
        
    @pytest.mark.unit
    def test_vocab_built_correctly(self):
        """Test that vocabulary was built from Jeeves text."""
        assert len(itos) > 0
        assert len(stoi) > 0
        assert len(itos) == len(stoi)
        
    @pytest.mark.unit
    def test_tokens_encoded(self):
        """Test that text was properly encoded to tokens."""
        assert isinstance(tokens, torch.Tensor)
        assert len(tokens) > 0
        assert tokens.dtype == torch.int64
        
    @pytest.mark.unit
    def test_dataset_created(self):
        """Test that CharDataset was created successfully."""
        assert len(dataset) > 0
        
        # Test getting a sample
        x, y = dataset[0]
        assert len(x) == 16
        assert len(y) == 16
        assert torch.equal(x[1:], y[:-1])  # y should be x shifted by 1