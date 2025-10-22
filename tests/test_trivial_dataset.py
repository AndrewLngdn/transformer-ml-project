import pytest
import torch
from src.trivial_dataset import TrivialDataset


class TestTrivialDataset:

    @pytest.mark.unit
    def test_sequence_shapes_and_dtype(self):
        """Test that sequences have correct shapes and dtype for embeddings."""
        dataset = TrivialDataset(seq_len=8)
        src, tgt = dataset[0]

        assert src.shape == (8,)
        assert tgt.shape == (8,)
        assert src.dtype == torch.long
        assert tgt.dtype == torch.long

    @pytest.mark.unit
    def test_vocab_range(self):
        """Test that all tokens are within valid vocab range."""
        dataset = TrivialDataset(vocab_size=5, dataset_len=10)

        for i in range(len(dataset)):
            src, tgt = dataset[i]
            assert torch.all(src >= 0)
            assert torch.all(src < dataset.vocab_size)
            assert torch.all(tgt >= 0)
            assert torch.all(tgt < dataset.vocab_size)

    @pytest.mark.unit
    def test_padding_is_consecutive(self):
        """Test that padding tokens (0) are consecutive at the end."""
        dataset = TrivialDataset(vocab_size=5, seq_len=8, dataset_len=10)

        for i in range(len(dataset)):
            src, tgt = dataset[i]
            if (src == 0).any():
                first_pad_idx = torch.where(src == 0)[0][0].item()
                assert torch.all(src[first_pad_idx:] == 0)

    @pytest.mark.unit
    def test_target_offset_from_source(self):
        """Test that target is offset from source for next-token prediction."""
        dataset = TrivialDataset(vocab_size=10, seq_len=6, dataset_len=1)
        src, tgt = dataset[0]

        # Target should be different from source (shifted prediction)
        assert not torch.all(src == tgt)

    @pytest.mark.unit
    def test_index_out_of_range(self):
        """Test that invalid indices raise IndexError."""
        dataset = TrivialDataset(dataset_len=5)

        with pytest.raises(IndexError):
            dataset[5]

        with pytest.raises(IndexError):
            dataset[-1]

    @pytest.mark.unit
    def test_varying_sequence_lengths(self):
        """Test that different indices produce different effective sequence lengths."""
        dataset = TrivialDataset(vocab_size=5, seq_len=10, dataset_len=10)

        seq_lengths = []
        for i in range(len(dataset)):
            src, tgt = dataset[i]
            non_pad_count = (src != 0).sum().item()
            seq_lengths.append(non_pad_count)

        # Should have varying lengths due to dynamic padding
        assert len(set(seq_lengths)) > 1

    @pytest.mark.unit
    def test_min_sequence_length_respected(self):
        """Test that seq_len_min parameter enforces minimum sequence length."""
        dataset = TrivialDataset(vocab_size=5, seq_len=10, dataset_len=10, seq_len_min=3)

        for i in range(len(dataset)):
            src, tgt = dataset[i]
            non_pad_count = (src != 0).sum().item()
            assert non_pad_count >= dataset.seq_len_min
