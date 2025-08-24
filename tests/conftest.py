import pytest
import torch


@pytest.fixture
def sample_text():
    """Simple text for testing."""
    return "Hello, world! This is a test."


@pytest.fixture 
def complex_text():
    """More complex text with various characters."""
    return "Jeeves (born Reginald Jeeves, nicknamed Reggie[1]) is a fictional character."


@pytest.fixture
def vocab_data(sample_text):
    """Pre-built vocabulary for testing."""
    from src.char_data import build_char_vocab
    itos, stoi = build_char_vocab(sample_text)
    return itos, stoi


@pytest.fixture
def encoded_tensor(sample_text, vocab_data):
    """Pre-encoded tensor for testing."""
    from src.char_data import encode
    itos, stoi = vocab_data
    return encode(sample_text, stoi)


@pytest.fixture
def device():
    """Get available device for testing (prioritizing MPS for Mac)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def small_model():
    """Create a small model for testing."""
    from src.hello_model import HelloModel
    return HelloModel(vocab_size=32, embedding_dim=16)