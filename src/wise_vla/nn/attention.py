import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float = 0):
        super().__init__()
        inner = ((int(8 * dim / 3) + 63) // 64) * 64
        self.up = nn.Linear(dim, 2 * inner, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


def rotary(x: Tensor) -> Tensor:
    """RoPE on action sequence only; graph nodes deliberately have no ordering."""
    length, dim = x.shape[-2:]
    inv = torch.exp(-math.log(10000) * torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = torch.arange(length, device=x.device).float()[:, None] * inv[None]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0, rope: bool = False):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.dropout, self.rope = dropout, rope

    def forward(
        self,
        query: Tensor,
        context: Tensor | None = None,
        mask: Tensor | None = None,
        bias: Tensor | None = None,
    ):
        context = query if context is None else context
        batch, qlen, dim = query.shape

        def split(x):
            return x.view(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = (
            self.q_norm(split(self.q(query))),
            self.k_norm(split(self.k(context))),
            split(self.v(context)),
        )
        if self.rope:
            q, k = rotary(q), rotary(k)
        attn_mask = bias
        if mask is not None:
            # All callers provide a global/null token, so no row is fully masked.
            padding = torch.zeros(batch, 1, 1, context.shape[1], device=q.device, dtype=q.dtype)
            padding.masked_fill_(~mask[:, None, None, :], float("-inf"))
            attn_mask = padding if bias is None else padding + bias.to(q.dtype)
        result = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(result.transpose(1, 2).reshape(batch, qlen, dim))


def masked_mean(x: Tensor, mask: Tensor):
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
