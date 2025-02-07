import torch.nn as nn
from src.multi_head_attention import MultiHeadAttention
from src.attention import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    def forward(self, src, mask=None):
        src, _ = self.self_attn(src, src, src, mask=mask)
        src = self.feed_forward(src)
        return src
