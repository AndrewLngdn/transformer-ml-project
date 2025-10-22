import pytest
import torch
from src.attention import masked_softmax, DotProductAttention, MultiHeadAttention, AdditiveAttention


class TestMaskedSoftmax:

    @pytest.mark.unit
    def test_no_masking(self):
        """Test that softmax works normally when no lengths provided."""
        X = torch.randn(2, 3, 4)
        result = masked_softmax(X, lengths=None)

        # Should sum to 1 along last dimension
        assert torch.allclose(result.sum(dim=-1), torch.ones(2, 3))
        assert result.shape == (2, 3, 4)

    @pytest.mark.unit
    def test_masking_with_1d_lengths(self):
        """Test masking with 1D lengths (B,) - same length for all heads."""
        B, H, T = 2, 3, 5
        X = torch.randn(B, H, T)
        lengths = torch.tensor([2, 4])  # (B,)

        result = masked_softmax(X, lengths)

        assert result.shape == (B, H, T)

        # First batch: only first 2 positions should be non-zero
        assert torch.all(result[0, :, :2] > 0)
        assert torch.allclose(result[0, :, 2:], torch.zeros(H, 3))

        # Second batch: only first 4 positions should be non-zero
        assert torch.all(result[1, :, :4] > 0)
        assert torch.allclose(result[1, :, 4:], torch.zeros(H, 1))

    @pytest.mark.unit
    def test_masking_with_2d_lengths(self):
        """Test masking with 2D lengths (B, H) - different length per head."""
        B, H, T = 2, 3, 5
        X = torch.randn(B, H, T)
        lengths = torch.tensor([[2, 3, 4], [1, 2, 5]])  # (B, H)

        result = masked_softmax(X, lengths)

        assert result.shape == (B, H, T)

        # First batch, first head: only first 2 positions
        assert torch.all(result[0, 0, :2] > 0)
        assert torch.allclose(result[0, 0, 2:], torch.zeros(3))

        # Second batch, third head: all 5 positions
        assert torch.all(result[1, 2, :5] > 0)

    @pytest.mark.unit
    def test_fully_masked_rows_are_zero(self):
        """Test that fully masked rows (length=0) result in all zeros."""
        B, H, T = 1, 2, 4
        X = torch.randn(B, H, T)
        lengths = torch.tensor([[0, 2]])  # (B, H)

        result = masked_softmax(X, lengths)

        # First head should be all zeros (length=0)
        assert torch.allclose(result[0, 0, :], torch.zeros(T))

        # Second head should have first 2 non-zero
        assert torch.all(result[0, 1, :2] > 0)

    @pytest.mark.unit
    def test_sums_to_one_for_valid_positions(self):
        """Test that probabilities sum to 1 for non-fully-masked rows."""
        X = torch.randn(2, 3, 5)
        lengths = torch.tensor([3, 5])

        result = masked_softmax(X, lengths)

        # Each row should sum to 1 (approximately)
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2, 3), atol=1e-6)


class TestDotProductAttention:

    @pytest.mark.unit
    def test_output_shape(self):
        """Test that output shape matches expected (batch_size, num_queries, value_dim)."""
        attn = DotProductAttention(dropout=0.0)
        attn.eval()

        batch_size, num_queries, num_kv = 2, 3, 5
        d = 4
        value_dim = 6

        queries = torch.randn(batch_size, num_queries, d)
        keys = torch.randn(batch_size, num_kv, d)
        values = torch.randn(batch_size, num_kv, value_dim)

        output = attn(queries, keys, values)

        assert output.shape == (batch_size, num_queries, value_dim)

    @pytest.mark.unit
    def test_attention_weights_stored(self):
        """Test that attention weights are stored after forward pass."""
        attn = DotProductAttention(dropout=0.0)
        attn.eval()

        queries = torch.randn(2, 3, 4)
        keys = torch.randn(2, 5, 4)
        values = torch.randn(2, 5, 6)

        output = attn(queries, keys, values)

        assert attn.attention_weights is not None
        assert attn.attention_weights.shape == (2, 3, 5)  # (batch, queries, keys)

    @pytest.mark.unit
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along key dimension."""
        attn = DotProductAttention(dropout=0.0)
        attn.eval()

        queries = torch.randn(2, 3, 4)
        keys = torch.randn(2, 5, 4)
        values = torch.randn(2, 5, 6)

        output = attn(queries, keys, values)

        weights_sum = attn.attention_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones(2, 3), atol=1e-6)

    @pytest.mark.unit
    def test_with_valid_lens_masking(self):
        """Test that valid_lens properly masks attention weights."""
        attn = DotProductAttention(dropout=0.0)
        attn.eval()

        queries = torch.randn(2, 3, 4)
        keys = torch.randn(2, 5, 4)
        values = torch.randn(2, 5, 6)
        valid_lens = torch.tensor([3, 5])  # First batch only attends to first 3 keys

        output = attn(queries, keys, values, valid_lens)

        # First batch should have zero weights for positions 3, 4
        assert torch.allclose(attn.attention_weights[0, :, 3:], torch.zeros(3, 2))

        # Second batch should attend to all 5 positions
        assert torch.all(attn.attention_weights[1, :, :] >= 0)


class TestMultiHeadAttention:

    @pytest.mark.unit
    def test_hidden_dim_divisibility(self):
        """Test that hidden_dim must be divisible by num_heads."""
        with pytest.raises(Exception, match="hidden dim must be divisible by num heads"):
            MultiHeadAttention(hidden_dim=32, num_heads=5, dropout=0.1)

    @pytest.mark.unit
    def test_output_shape(self):
        """Test that output shape is (batch_size, num_queries, hidden_dim)."""
        mha = MultiHeadAttention(hidden_dim=32, num_heads=4, dropout=0.0)
        mha.eval()

        batch_size, num_queries, num_kv = 2, 6, 8

        queries = torch.randn(batch_size, num_queries, 32)
        keys = torch.randn(batch_size, num_kv, 32)
        values = torch.randn(batch_size, num_kv, 32)

        output = mha(queries, keys, values, valid_lens=None)

        assert output.shape == (batch_size, num_queries, 32)

    @pytest.mark.unit
    def test_self_attention(self):
        """Test self-attention where Q, K, V are the same."""
        mha = MultiHeadAttention(hidden_dim=16, num_heads=2, dropout=0.0)
        mha.eval()

        X = torch.randn(2, 5, 16)

        output = mha(X, X, X, valid_lens=None)

        assert output.shape == (2, 5, 16)

    @pytest.mark.unit
    def test_cross_attention_different_lengths(self):
        """Test cross-attention where queries and keys have different lengths."""
        mha = MultiHeadAttention(hidden_dim=24, num_heads=3, dropout=0.0)
        mha.eval()

        queries = torch.randn(2, 4, 24)  # 4 queries
        keys = torch.randn(2, 10, 24)     # 10 key-value pairs
        values = torch.randn(2, 10, 24)

        output = mha(queries, keys, values, valid_lens=None)

        assert output.shape == (2, 4, 24)

    @pytest.mark.unit
    def test_with_valid_lens(self):
        """Test that valid_lens masking works correctly."""
        mha = MultiHeadAttention(hidden_dim=16, num_heads=2, dropout=0.0)
        mha.eval()

        batch_size = 2
        queries = torch.randn(batch_size, 3, 16)
        keys = torch.randn(batch_size, 5, 16)
        values = torch.randn(batch_size, 5, 16)
        valid_lens = torch.tensor([2, 4])

        output = mha(queries, keys, values, valid_lens)

        assert output.shape == (batch_size, 3, 16)
        # Output should be different from unmasked version
        output_unmasked = mha(queries, keys, values, None)
        assert not torch.allclose(output, output_unmasked)

    @pytest.mark.unit
    def test_attention_weights_shape(self):
        """Test that stored attention weights have correct shape."""
        mha = MultiHeadAttention(hidden_dim=16, num_heads=4, dropout=0.0)
        mha.eval()

        batch_size, num_queries, num_kv = 2, 3, 5
        queries = torch.randn(batch_size, num_queries, 16)
        keys = torch.randn(batch_size, num_kv, 16)
        values = torch.randn(batch_size, num_kv, 16)

        output = mha(queries, keys, values, None)

        # Attention weights are stored in the underlying DotProductAttention
        # Shape should be (batch_size * num_heads, num_queries, num_kv)
        expected_shape = (batch_size * 4, num_queries, num_kv)
        assert mha.attention.attention_weights.shape == expected_shape

    @pytest.mark.unit
    def test_no_bias_option(self):
        """Test that bias parameter is respected."""
        mha_no_bias = MultiHeadAttention(hidden_dim=16, num_heads=2, dropout=0.0, bias=False)
        mha_with_bias = MultiHeadAttention(hidden_dim=16, num_heads=2, dropout=0.0, bias=True)

        assert mha_no_bias.W_q.bias is None
        assert mha_with_bias.W_q.bias is not None


class TestAdditiveAttention:

    @pytest.mark.unit
    def test_output_shape(self):
        """Test that output shape matches expected (batch_size, num_queries, value_dim)."""
        attn = AdditiveAttention(num_hiddens=8, dropout=0.0)
        attn.eval()

        batch_size, num_queries, num_kv = 2, 3, 5
        value_dim = 4

        queries = torch.randn(batch_size, num_queries, 20)
        keys = torch.randn(batch_size, num_kv, 2)
        values = torch.randn(batch_size, num_kv, value_dim)

        output = attn(queries, keys, values)

        assert output.shape == (batch_size, num_queries, value_dim)

    @pytest.mark.unit
    def test_attention_scores_stored(self):
        """Test that attention scores are stored after forward pass."""
        attn = AdditiveAttention(num_hiddens=8, dropout=0.0)
        attn.eval()

        queries = torch.randn(2, 3, 20)
        keys = torch.randn(2, 5, 2)
        values = torch.randn(2, 5, 4)

        output = attn(queries, keys, values)

        assert hasattr(attn, 'attention_scores')
        assert attn.attention_scores is not None
        assert attn.attention_scores.shape == (2, 3, 5)

    @pytest.mark.unit
    def test_with_valid_lens(self):
        """Test that valid_lens masking works correctly."""
        attn = AdditiveAttention(num_hiddens=8, dropout=0.0)
        attn.eval()

        queries = torch.randn(2, 3, 20)
        keys = torch.randn(2, 5, 2)
        values = torch.randn(2, 5, 4)
        valid_lens = torch.tensor([2, 4])

        output = attn(queries, keys, values, valid_lens)

        assert output.shape == (2, 3, 4)

        # First batch should have zero scores for positions 2, 3, 4
        assert torch.allclose(attn.attention_scores[0, :, 2:], torch.zeros(3, 3))

    @pytest.mark.unit
    def test_scores_sum_to_one(self):
        """Test that attention scores sum to 1 along key dimension."""
        attn = AdditiveAttention(num_hiddens=8, dropout=0.0)
        attn.eval()

        queries = torch.randn(2, 3, 20)
        keys = torch.randn(2, 5, 2)
        values = torch.randn(2, 5, 4)

        output = attn(queries, keys, values)

        scores_sum = attn.attention_scores.sum(dim=-1)
        assert torch.allclose(scores_sum, torch.ones(2, 3), atol=1e-6)


class TestAttentionComparison:

    @pytest.mark.integration
    def test_dot_product_vs_additive_different_outputs(self):
        """Test that dot product and additive attention produce different results."""
        batch_size, num_queries, num_kv = 2, 3, 5
        d = 4
        value_dim = 6

        queries = torch.randn(batch_size, num_queries, d)
        keys = torch.randn(batch_size, num_kv, d)
        values = torch.randn(batch_size, num_kv, value_dim)

        dot_attn = DotProductAttention(dropout=0.0)
        add_attn = AdditiveAttention(num_hiddens=8, dropout=0.0)

        dot_attn.eval()
        add_attn.eval()

        dot_output = dot_attn(queries, keys, values)
        add_output = add_attn(queries, keys, values)

        # Both should have same shape
        assert dot_output.shape == add_output.shape

        # But different values (different attention mechanisms)
        assert not torch.allclose(dot_output, add_output, atol=0.1)

    @pytest.mark.integration
    def test_multihead_wraps_dotproduct(self):
        """Test that MultiHeadAttention uses DotProductAttention internally."""
        mha = MultiHeadAttention(hidden_dim=16, num_heads=2, dropout=0.0)

        assert isinstance(mha.attention, DotProductAttention)
