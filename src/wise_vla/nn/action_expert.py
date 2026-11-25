import math

import torch
from torch import nn

from .attention import Attention, RMSNorm, SwiGLU


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        phase = t.float()[:, None] * 1000 * freq[None]
        return self.mlp(torch.cat((phase.sin(), phase.cos()), -1))


class ActionBlock(nn.Module):
    """Bidirectional action attention with separate VLM and graph cross-attention."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(4)])
        self.self_attn = Attention(dim, heads, dropout, rope=True)
        self.vlm_attn = Attention(dim, heads, dropout)
        self.graph_attn = Attention(dim, heads, dropout)
        self.ff = SwiGLU(dim, dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.vlm_gate = nn.Parameter(torch.tensor(-1.0))
        self.graph_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, condition, visual, visual_mask, graph, graph_mask, action_mask):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(condition).chunk(6, -1)
        normalized = self.norms[0](x) * (1 + scale_a[:, None]) + shift_a[:, None]
        x = x + gate_a[:, None].sigmoid() * self.self_attn(normalized, mask=action_mask)
        x = x + self.vlm_gate.sigmoid() * self.vlm_attn(self.norms[1](x), visual, visual_mask)
        x = x + self.graph_gate.sigmoid() * self.graph_attn(self.norms[2](x), graph, graph_mask)
        normalized = self.norms[3](x) * (1 + scale_f[:, None]) + shift_f[:, None]
        x = x + gate_f[:, None].sigmoid() * self.ff(normalized)
        return x * action_mask.unsqueeze(-1)


class ActionExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, cfg = config.hidden_dim, config.action
        self.action_in = nn.Linear(cfg.dim, dim)
        self.state_in = nn.Sequential(nn.Linear(cfg.state_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = TimeEmbedding(dim)
        self.blocks = nn.ModuleList(
            [ActionBlock(dim, config.num_heads, config.dropout) for _ in range(cfg.layers)]
        )
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, cfg.dim)

    def forward(self, noisy_actions, t, state, context, valid):
        x = self.action_in(torch.where(valid, noisy_actions, 0))
        condition = self.time(t) + self.state_in(state)
        if context.target_condition is not None:
            condition = condition + context.target_condition
        # Preserve a direct conditioning path even at zero-initialized AdaLN modulation.
        x = x + condition[:, None]
        step_mask = valid.any(-1)
        for block in self.blocks:
            x = block(
                x,
                condition,
                context.visual,
                context.visual_mask,
                context.graph,
                context.graph_mask,
                step_mask,
            )
        return self.out(self.norm(x)).masked_fill(~valid, 0)
