import torch.nn as nn
from src.encoder_layer import EncoderLayer
from src.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.position = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, src, mask=None):
        src = self.position(src)
        for layer in self.layers:
            src = layer(src, mask=mask)
        return self.norm(src)