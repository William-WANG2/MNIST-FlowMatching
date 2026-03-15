import math

import torch
from torch import nn


class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("FourierEncoder expects an even embedding dimension.")
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        freqs = t * self.weights * 2.0 * math.pi
        sin_embed = torch.sin(freqs)
        cos_embed = torch.cos(freqs)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2.0)

