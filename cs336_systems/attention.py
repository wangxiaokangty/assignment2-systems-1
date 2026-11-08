"""Explicit scaled dot-product attention and memory-efficient tiled attention."""
import torch


def attention(q, k, v, is_causal=False):
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if is_causal:
        mask = torch.arange(q.shape[-2], device=q.device)[:, None] < torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(mask, float('-inf'))
    return scores.softmax(-1) @ v


