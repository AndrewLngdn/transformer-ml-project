import pytest
import torch
from src.rnn_model import NextCharRNN


class TestNextCharRNN:
    """Smoke tests for NextCharRNN model."""
    
    @pytest.mark.unit
    def test_rnn_model_creates(self):
        """Test that RNN model can be created."""
        model = NextCharRNN(vocab_size=32, embedding_dim=16, hidden_size=32)
        
        assert model.vocab_size == 32
        assert model.embedding_dim == 16
        
    @pytest.mark.unit  
    def test_rnn_forward_pass(self):
        """Test that RNN model can perform forward pass."""
        model = NextCharRNN(vocab_size=32, embedding_dim=16, hidden_size=32)
        
        # Test with sequence input
        batch_size = 2
        seq_length = 10
        x = torch.randint(0, 32, (batch_size, seq_length))
        
        output = model(x)
        
        # Output should be same shape as input but with vocab_size in last dim
        expected_shape = (batch_size, seq_length, 32)
        assert output.shape == expected_shape
        assert output.dtype == torch.float32
        
    @pytest.mark.unit
    def test_rnn_with_dropout(self):
        """Test RNN model with dropout."""
        model = NextCharRNN(vocab_size=32, embedding_dim=16, hidden_size=32, dropout=0.1)
        x = torch.randint(0, 32, (1, 5))
        
        output = model(x)
        assert output.shape == (1, 5, 32)