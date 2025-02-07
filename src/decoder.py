import torch.nn as nn
from src.decoder_layer import DecoderLayer
from src.positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.position = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        tgt = self.position(tgt)
        attention_weights = []
        for layer in self.layers:
            tgt, attn = layer(tgt, src, tgt_mask=tgt_mask, src_mask=src_mask)
            attention_weights.append(attn)
        return self.norm(tgt), attention_weights