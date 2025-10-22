import pytest
import torch
from src.positional_encoding import LazySinCosPositionalEncoding


class TestLazySinCosPositionalEncoding:

    @pytest.mark.unit
    def test_creates_and_runs(self):
        """Test that positional encoding can be created and run a forward pass."""
        pe = LazySinCosPositionalEncoding(dropout=0.1, embed_dim=16)
        x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, embed_dim=16

        output = pe(x)

        assert output.shape == (2, 5, 16)
        assert output.dtype == torch.float32

    @pytest.mark.unit
    def test_output_shape(self):
        """Test that output shape matches input shape exactly."""
        pe = LazySinCosPositionalEncoding(dropout=0.0, embed_dim=32)

        # Test different shapes
        for batch_size in [1, 3, 8]:
            for seq_len in [1, 10, 50]:
                x = torch.randn(batch_size, seq_len, 32)
                output = pe(x)
                assert output.shape == x.shape

    @pytest.mark.unit
    def test_lazy_initialization(self):
        """Test that P matrix is created lazily on first forward pass."""
        pe = LazySinCosPositionalEncoding(dropout=0.1, embed_dim=16)

        # Initially P should be None
        assert pe.P is None

        # After forward pass, P should be created
        x = torch.randn(1, 5, 16)
        pe(x)
        assert pe.P is not None
        assert pe.P.shape == (1000, 16)  # max_len x embed_dim

    @pytest.mark.unit
    def test_device_compatibility(self, device):
        """Test that positional encoding works on available device."""
        pe = LazySinCosPositionalEncoding(dropout=0.1, embed_dim=16).to(device)
        x = torch.randn(2, 5, 16, device=device)

        output = pe(x)

        assert output.device.type == device.type
        assert pe.P.device.type == device.type

    @pytest.mark.unit
    def test_sincos_pattern(self):
        """Test that even dimensions use sin and odd dimensions use cos."""
        pe = LazySinCosPositionalEncoding(dropout=0.0, embed_dim=4)
        x = torch.zeros(1, 2, 4)  # Use zeros to isolate positional encoding

        output = pe(x)

        # The pattern should be sin, cos, sin, cos for dimensions 0, 1, 2, 3
        # For position 0, sin should be 0 and cos should be 1
        pos_0 = output[0, 0, :]  # First position

        # Check that pattern alternates (approximate due to floating point)
        assert abs(pos_0[0].item() - 0.0) < 1e-6  # sin(0) = 0
        assert abs(pos_0[1].item() - 1.0) < 1e-6  # cos(0) = 1
        assert abs(pos_0[2].item() - 0.0) < 1e-6  # sin(0) = 0
        assert abs(pos_0[3].item() - 1.0) < 1e-6  # cos(0) = 1