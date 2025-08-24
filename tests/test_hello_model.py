import pytest
import torch
from src.hello_model import HelloModel


class TestHelloModel:
    """Smoke tests for HelloModel."""
    
    @pytest.mark.unit
    def test_model_creates_and_runs(self):
        """Test that model can be created and run a forward pass."""
        model = HelloModel(vocab_size=32, embedding_dim=16)
        x = torch.randint(0, 32, (5,))
        
        output = model(x)
        
        assert output.shape == (5, 32)
        assert output.dtype == torch.float32