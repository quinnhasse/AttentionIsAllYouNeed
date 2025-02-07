import torch

def make_src_mask(src, pad_idx):

    mask = (src != pad_idx).unsqueeze(-2)
    return mask 

def make_tgt_mask(tgt, pad_idx):

    tgt_mask = (tgt != pad_idx).unsqueeze(-2)
    tgt_len = tgt.size(1)
    subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    tgt_mask = tgt_mask & subsequent_mask
    return tgt_mask