import torch
from torch import nn
from torch.nn import functional as F

from .attention import Attention, RMSNorm, SwiGLU, masked_mean


class RelationBlock(nn.Module):
    def __init__(self, dim, heads, relations, dropout):
        super().__init__()
        self.forward_bias = nn.Parameter(torch.randn(relations, heads) * 0.02)
        self.reverse_bias = nn.Parameter(torch.randn(relations, heads) * 0.02)
        self.outgoing = nn.Parameter(torch.randn(relations, dim) * 0.02)
        self.incoming = nn.Parameter(torch.randn(relations, dim) * 0.02)
        self.self_attn = Attention(dim, heads, dropout)
        self.visual_attn = Attention(dim, heads, dropout)
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(3)])
        self.mlp = SwiGLU(dim, dropout)
        self.visual_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x, edges, mask, visual, visual_mask):
        bias = torch.einsum("bijr,rh->bhij", edges, self.forward_bias)
        bias = bias + torch.einsum("bjir,rh->bhij", edges, self.reverse_bias)
        bias = F.pad(bias, (1, 0, 1, 0))  # learned global token has no geometric edge
        outgoing, incoming = edges.sum(2), edges.sum(1)
        messages = outgoing @ self.outgoing + incoming @ self.incoming
        messages = messages / (outgoing.sum(-1) + incoming.sum(-1)).clamp_min(1).unsqueeze(-1)
        x = x + F.pad(messages, (0, 0, 1, 0))
        x = x + self.self_attn(self.norms[0](x), mask=mask, bias=bias)
        x = x + self.visual_gate.sigmoid() * self.visual_attn(self.norms[1](x), visual, visual_mask)
        x = x + self.mlp(self.norms[2](x))
        return x * mask.unsqueeze(-1)


class GraphEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, cfg = config.hidden_dim, config.graph
        self.features = nn.Sequential(
            nn.Linear(cfg.node_feature_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.types = nn.Embedding(cfg.num_node_types, dim)
        self.global_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                RelationBlock(dim, config.num_heads, len(cfg.relations), config.dropout)
                for _ in range(cfg.layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.target_query, self.target_key = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.relation_source, self.relation_target = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.relation_head = nn.Sequential(nn.SiLU(), nn.Linear(dim, len(cfg.relations)))

    def forward(self, graph, semantics, visual, visual_mask):
        nodes = self.features(graph.features) + self.types(graph.node_types) + semantics
        nodes = nodes * graph.node_mask.unsqueeze(-1)
        global_token = self.global_token.expand(nodes.shape[0], -1, -1)
        x = torch.cat((global_token, nodes), dim=1)
        mask = F.pad(graph.node_mask, (1, 0), value=True)
        for block in self.blocks:
            x = block(x, graph.edges, mask, visual, visual_mask)
        return self.norm(x) * mask.unsqueeze(-1), mask

    def target_logits(self, graph_tokens, graph_mask, visual, visual_mask):
        query = self.target_query(masked_mean(visual, visual_mask))
        keys = self.target_key(graph_tokens[:, 1:])
        logits = torch.einsum("bd,bnd->bn", query, keys) / keys.shape[-1] ** 0.5
        return logits.masked_fill(~graph_mask[:, 1:], torch.finfo(logits.dtype).min)

    def relation_logits(self, graph_tokens):
        nodes = graph_tokens[:, 1:]
        return self.relation_head(
            self.relation_source(nodes).unsqueeze(2) + self.relation_target(nodes).unsqueeze(1)
        )
