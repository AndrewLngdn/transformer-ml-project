from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, enc_X, dec_X, seq_lens=None, *args):
        enc_all_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        
        return self.decoder(dec_X, dec_state, seq_lens)[0]
        