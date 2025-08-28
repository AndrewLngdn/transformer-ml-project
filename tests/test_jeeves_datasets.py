import pytest
import torch
from unittest.mock import patch, mock_open


class TestJeevesDatasets:
    """Smoke tests for Jeeves dataset functionality."""
    
    @pytest.mark.unit
    @patch('src.jeeves_datasets.read_txt')
    def test_datasets_created_with_mock_data(self, mock_read_txt):
        """Test that datasets are created when data files are available."""
        # Mock the file reading to return test data
        mock_read_txt.side_effect = [
            "My Man Jeeves test content.",
            "Right Ho Jeeves test content.", 
            "The Inimitable Jeeves test content."
        ]
        
        # Import after mocking to ensure mock takes effect
        from src import jeeves_datasets
        
        # Force reload to get fresh data with mocks
        import importlib
        importlib.reload(jeeves_datasets)
        
        # Test that datasets exist
        assert hasattr(jeeves_datasets, 'train_dataset')
        assert hasattr(jeeves_datasets, 'val_dataset') 
        assert hasattr(jeeves_datasets, 'test_dataset')
        
        # Test basic functionality if datasets were created
        if len(jeeves_datasets.train_dataset) > 0:
            x, y = jeeves_datasets.train_dataset[0]
            assert len(x) == 64  # Default T is now 64
            assert len(y) == 64
    
    @pytest.mark.unit
    def test_vocab_structure(self):
        """Test that vocabulary has expected structure when created."""
        # This tests the basic import without requiring data files
        try:
            from src.jeeves_datasets import itos, stoi
            
            # If successfully imported, test structure
            assert len(itos) == len(stoi)
            if len(itos) > 0:
                # Test that itos and stoi are consistent
                for i, char in enumerate(itos):
                    assert stoi[char] == i
        except (FileNotFoundError, ImportError):
            # Expected if data files don't exist
            pytest.skip("Data files not available for testing")
            
    @pytest.mark.unit
    @patch('src.jeeves_datasets.read_txt')
    def test_jeeves_datasets_class(self, mock_read_txt):
        """Test the new JeevesDatasets class with custom T."""
        # Mock the file reading
        mock_read_txt.side_effect = [
            "Test content for my man.",
            "Test content for right ho.",
            "Test content for inimitable."
        ]
        
        from src.jeeves_datasets import JeevesDatasets
        
        # Test with custom T
        datasets = JeevesDatasets(T=32)
        
        assert datasets.T == 32
        assert datasets.train_frac == 0.8
        assert datasets.vocab_size > 0
        
        # Test datasets exist and have correct T
        x, y = datasets.train_dataset[0]
        assert len(x) == 32
        assert len(y) == 32