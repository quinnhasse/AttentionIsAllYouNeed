import torch.nn as nn
import torch
from src.encoder import Encoder
from src.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, d_inner=2048, n_layers=6,
                 n_head=8, d_k=64, d_v=64, dropout=0.1, max_seq_len=100):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.encoder = Encoder(n_layers, d_model, d_inner, n_head, d_k, d_v, dropout, max_seq_len)
        self.decoder = Decoder(n_layers, d_model, d_inner, n_head, d_k, d_v, dropout, max_seq_len)
        self.out = nn.Linear(d_model, tgt_vocab)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_src = self.encoder(self.src_emb(src), mask=src_mask)
        dec_tgt = self.decoder(self.tgt_emb(tgt), enc_src, tgt_mask=tgt_mask, src_mask=src_mask)
        output = self.out(dec_tgt[0])
        return output