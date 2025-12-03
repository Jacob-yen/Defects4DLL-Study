import torch
import torch.nn as nn


with torch.no_grad():
    B = 8
    L = 100
    D = 512
    H = 8

    mha = nn.MultiheadAttention(D, H, batch_first=True)

    X = torch.randn(B, L, D)
    M = torch.randn(B * H, L, L) > 0

    mha.eval()  # enable fast path
    mha(X, X, X, attn_mask=M, need_weights=False)  # crashes