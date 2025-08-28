import pytest
import tempfile
import os
from src.data_importer import read_txt


class TestDataImporter:
    """Smoke tests for data_importer functionality."""
    
    @pytest.mark.unit
    def test_read_txt_basic(self):
        """Test that read_txt can read a basic text file."""
        test_content = "Hello, world!\nThis is a test file."
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write(test_content)
            temp_filename = f.name
            
        try:
            # Read the file using our function
            result = read_txt(temp_filename)
            assert result == test_content
        finally:
            # Clean up
            os.unlink(temp_filename)
            
    @pytest.mark.unit
    def test_read_txt_with_encoding(self):
        """Test that read_txt handles UTF-8 with BOM correctly."""
        test_content = "Hello, world! 🌟 Test with emoji and special chars: áéíóú"
        
        # Create a temporary file with UTF-8-BOM encoding
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8-sig') as f:
            f.write(test_content)
            temp_filename = f.name
            
        try:
            # Read the file using our function
            result = read_txt(temp_filename)
            assert result == test_content
        finally:
            # Clean up
            os.unlink(temp_filename)
            
    @pytest.mark.unit
    def test_read_txt_empty_file(self):
        """Test reading an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            temp_filename = f.name
            
        try:
            result = read_txt(temp_filename)
            assert result == ""
        finally:
            os.unlink(temp_filename)