import pytest
import torch
from src.transformer import (
    PositionWiseFF, AddNorm, TransformerEncoderBlock, TransformerEncoder,
    TransformerDecoderBlock, TransformerDecoder
)


class TestTransformerComponents:

    def test_position_wise_ff_creates_and_runs(self):
        """Test that PositionWiseFF can be created and run a forward pass."""
        ffn = PositionWiseFF(ffn_num_hiddens=64, ffn_num_outputs=32)
        x = torch.randn(2, 5, 32)  # batch=2, seq_len=5, embed_dim=32

        output = ffn(x)

        assert output.shape == (2, 5, 32)
        assert output.dtype == torch.float32

    def test_add_norm_creates_and_runs(self):
        """Test that AddNorm can be created and run a forward pass."""
        add_norm = AddNorm(norm_shape=32, dropout=0.1)
        x = torch.randn(2, 5, 32)
        y = torch.randn(2, 5, 32)

        output = add_norm(x, y)

        assert output.shape == (2, 5, 32)
        assert output.dtype == torch.float32

    def test_transformer_encoder_block_creates_and_runs(self):
        """Test that TransformerEncoderBlock can be created and run a forward pass."""
        block = TransformerEncoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_num_hiddens=64,
            dropout=0.1
        )
        x = torch.randn(2, 5, 32)  # batch=2, seq_len=5, embed_dim=32
        valid_lens = None

        output = block(x, valid_lens)

        assert output.shape == (2, 5, 32)
        assert output.dtype == torch.float32

    def test_transformer_encoder_creates_and_runs(self):
        """Test that TransformerEncoder can be created and run a forward pass."""
        encoder = TransformerEncoder(
            vocab_size=100,
            embed_dim=32,
            num_heads=4,
            num_blocks=2,
            dropout=0.1
        )
        x = torch.randint(0, 100, (2, 5))  # batch=2, seq_len=5, token_ids
        valid_lens = None

        output = encoder(x, valid_lens)

        assert output.shape == (2, 5, 32)
        assert output.dtype == torch.float32

    def test_transformer_encoder_output_shape(self):
        """Test that TransformerEncoder output shape is correct for different inputs."""
        encoder = TransformerEncoder(
            vocab_size=50,
            embed_dim=16,
            num_heads=2,
            num_blocks=1,
            dropout=0.0
        )

        # Test different batch sizes and sequence lengths
        for batch_size in [1, 3]:
            for seq_len in [1, 8, 20]:
                x = torch.randint(0, 50, (batch_size, seq_len))
                output = encoder(x, None)
                assert output.shape == (batch_size, seq_len, 16)


class TestTransformerDecoder:

    @pytest.mark.unit
    def test_decoder_block_forward_training(self):
        """Test TransformerDecoderBlock in training mode."""
        block = TransformerDecoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_num_hiddens=64,
            dropout=0.1,
            i=0
        )
        block.train()

        batch_size, seq_len = 2, 5
        X = torch.randn(batch_size, seq_len, 32)
        enc_output = torch.randn(batch_size, 8, 32)
        enc_valid_lens = None
        state = [enc_output, enc_valid_lens, [None, None]]

        output, new_state = block(X, state)

        assert output.shape == (batch_size, seq_len, 32)
        assert new_state[2][0] is not None  # Cache should be updated

    @pytest.mark.unit
    def test_decoder_block_forward_eval(self):
        """Test TransformerDecoderBlock in eval mode (autoregressive)."""
        block = TransformerDecoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_num_hiddens=64,
            dropout=0.0,
            i=0
        )
        block.eval()

        batch_size = 2
        X = torch.randn(batch_size, 1, 32)  # Single token
        enc_output = torch.randn(batch_size, 8, 32)
        enc_valid_lens = None
        state = [enc_output, enc_valid_lens, [None, None]]

        output, new_state = block(X, state)

        assert output.shape == (batch_size, 1, 32)
        assert new_state[2][0].shape == (batch_size, 1, 32)

    @pytest.mark.unit
    def test_decoder_block_caching(self):
        """Test that decoder block properly caches previous inputs."""
        block = TransformerDecoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_num_hiddens=64,
            dropout=0.0,
            i=0
        )
        block.eval()

        batch_size = 1
        enc_output = torch.randn(batch_size, 5, 32)
        state = [enc_output, None, [None, None]]

        # First token
        X1 = torch.randn(batch_size, 1, 32)
        output1, state = block(X1, state)
        assert state[2][0].shape == (batch_size, 1, 32)

        # Second token - cache should grow
        X2 = torch.randn(batch_size, 1, 32)
        output2, state = block(X2, state)
        assert state[2][0].shape == (batch_size, 2, 32)

    @pytest.mark.unit
    def test_decoder_init_state(self):
        """Test that TransformerDecoder.init_state creates proper state structure."""
        decoder = TransformerDecoder(
            vocab_size=100,
            embed_dim=32,
            num_heads=4,
            num_blocks=2,
            dropout=0.1
        )

        enc_outputs = torch.randn(2, 8, 32)
        enc_valid_lens = torch.tensor([5, 8])

        state = decoder.init_state(enc_outputs, enc_valid_lens)

        assert len(state) == 3
        assert state[0].shape == (2, 8, 32)
        assert torch.equal(state[1], enc_valid_lens)
        assert len(state[2]) == 2  # num_blocks
        assert all(x is None for x in state[2])

    @pytest.mark.unit
    def test_decoder_forward_output_shape(self):
        """Test that TransformerDecoder produces correct output shape."""
        decoder = TransformerDecoder(
            vocab_size=100,
            embed_dim=32,
            num_heads=4,
            num_blocks=2,
            dropout=0.1
        )

        batch_size, tgt_len = 2, 6
        X = torch.randint(0, 100, (batch_size, tgt_len))
        enc_outputs = torch.randn(batch_size, 8, 32)
        state = decoder.init_state(enc_outputs, None)

        output, new_state = decoder(X, state)

        assert output.shape == (batch_size, tgt_len, 100)  # vocab_size
        assert output.dtype == torch.float32

    @pytest.mark.unit
    def test_decoder_attention_weights_tracked(self):
        """Test that decoder tracks attention weights."""
        decoder = TransformerDecoder(
            vocab_size=100,
            embed_dim=32,
            num_heads=4,
            num_blocks=2,
            dropout=0.0
        )

        batch_size, tgt_len = 2, 6
        X = torch.randint(0, 100, (batch_size, tgt_len))
        enc_outputs = torch.randn(batch_size, 8, 32)
        state = decoder.init_state(enc_outputs, None)

        output, _ = decoder(X, state)

        # Check attention weights are stored
        attn_weights = decoder.attention_weights
        assert len(attn_weights) == 2  # [self_attn, enc_dec_attn]
        assert len(attn_weights[0]) == 2  # num_blocks
        assert len(attn_weights[1]) == 2  # num_blocks


class TestEncoderDecoderIntegration:

    @pytest.mark.integration
    def test_encoder_decoder_forward_pass(self):
        """Test full encoder-decoder forward pass."""
        vocab_size = 50
        embed_dim = 32
        num_heads = 4
        num_blocks = 2

        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=0.1
        )

        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=0.1
        )

        batch_size = 2
        src_len, tgt_len = 8, 6

        # Encoder forward
        src = torch.randint(0, vocab_size, (batch_size, src_len))
        enc_outputs = encoder(src, None)
        assert enc_outputs.shape == (batch_size, src_len, embed_dim)

        # Decoder forward
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
        state = decoder.init_state(enc_outputs, None)
        dec_outputs, _ = decoder(tgt, state)
        assert dec_outputs.shape == (batch_size, tgt_len, vocab_size)

    @pytest.mark.integration
    def test_encoder_decoder_with_valid_lens(self):
        """Test encoder-decoder with variable length sequences."""
        vocab_size = 50
        embed_dim = 32

        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=4,
            num_blocks=1,
            dropout=0.0
        )

        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=4,
            num_blocks=1,
            dropout=0.0
        )

        batch_size = 3
        src_len, tgt_len = 10, 8

        src = torch.randint(1, vocab_size, (batch_size, src_len))
        tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
        src_valid_lens = torch.tensor([5, 8, 10])

        # Encode
        enc_outputs = encoder(src, src_valid_lens)

        # Decode
        state = decoder.init_state(enc_outputs, src_valid_lens)
        dec_outputs, _ = decoder(tgt, state)

        assert dec_outputs.shape == (batch_size, tgt_len, vocab_size)

    @pytest.mark.integration
    def test_autoregressive_decoding(self):
        """Test autoregressive decoding one token at a time."""
        vocab_size = 30
        embed_dim = 16

        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=2,
            num_blocks=1,
            dropout=0.0
        )

        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=2,
            num_blocks=1,
            dropout=0.0
        )
        decoder.eval()

        batch_size = 1
        src = torch.randint(1, vocab_size, (batch_size, 5))

        # Encode
        enc_outputs = encoder(src, None)
        state = decoder.init_state(enc_outputs, None)

        # Decode autoregressively
        outputs = []
        for i in range(3):
            token = torch.randint(1, vocab_size, (batch_size, 1))
            output, state = decoder(token, state)
            outputs.append(output)

            # Verify cache grows
            assert state[2][0].shape[1] == i + 1

        # All outputs should have shape (batch_size, 1, vocab_size)
        for output in outputs:
            assert output.shape == (batch_size, 1, vocab_size)