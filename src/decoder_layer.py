import torch.nn as nn
from src.multi_head_attention import MultiHeadAttention
from src.attention import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.src_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        tgt, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt, attention = self.src_attn(tgt, src, src, mask=src_mask)
        tgt = self.feed_forward(tgt)
        return tgt, attention