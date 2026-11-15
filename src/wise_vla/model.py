"""Full differentiable WISE policy: pretrained VLM -> graph -> continuous actions."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import build_backbone
from .config import WISEConfig
from .graph import GraphBatch
from .nn.action_expert import ActionExpert
from .nn.attention import RMSNorm
from .nn.graph_encoder import GraphEncoder


@dataclass
class PolicyContext:
    visual: Tensor
    visual_mask: Tensor
    graph: Tensor
    graph_mask: Tensor
    target_logits: Tensor


class WISEVLA(nn.Module):
    def __init__(self, config: WISEConfig, backbone=None):
        super().__init__()
        self.config = config
        self.backbone = backbone if backbone is not None else build_backbone(config.vlm)
        if config.vlm.backend == "qwen3_vl":
            resolved = getattr(self.backbone.model.config, "_commit_hash", None)
            if resolved:
                config.vlm.revision = resolved
        dim = config.hidden_dim
        self.visual_projection = nn.Sequential(
            nn.Linear(self.backbone.output_dim, dim), RMSNorm(dim)
        )
        self.semantic_projection = nn.Sequential(
            nn.Linear(self.backbone.output_dim, dim), RMSNorm(dim)
        )
        self.graph_encoder = GraphEncoder(config)
        self.action_expert = ActionExpert(config)

    def encode(self, vlm_inputs: dict, graph: GraphBatch) -> PolicyContext:
        features = self.backbone(vlm_inputs)
        dtype = self.visual_projection[0].weight.dtype
        visual = self.visual_projection(features.tokens.to(dtype))
        semantics = self.semantic_projection(features.node_semantics.to(dtype))
        nodes, mask = self.graph_encoder(graph, semantics, visual, features.mask)
        targets = self.graph_encoder.target_logits(nodes, mask, visual, features.mask)
        return PolicyContext(visual, features.mask, nodes, mask, targets)

    def _valid_actions(self, actions, mask):
        expected = (actions.shape[0], self.config.action.horizon, self.config.action.dim)
        if actions.shape != expected:
            raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
        if mask is None:
            mask = torch.ones_like(actions, dtype=torch.bool)
        elif mask.ndim == 2:
            mask = mask[..., None].expand_as(actions)
        if mask.shape != actions.shape or mask.dtype != torch.bool:
            raise ValueError("action_mask must be bool [B,H] or [B,H,A]")
        if not mask.flatten(1).any(1).all():
            raise ValueError("Every sample needs at least one valid action")
        if not torch.isfinite(actions[mask]).all():
            raise ValueError("Non-finite valid actions")
        return mask

    def forward(
        self,
        vlm_inputs: dict,
        graph: GraphBatch,
        state: Tensor,
        actions: Tensor,
        action_mask: Tensor | None = None,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        mask_relations: bool = True,
    ):
        valid = self._valid_actions(actions, action_mask)
        b = actions.shape[0]
        if state.shape != (b, self.config.action.state_dim) or not torch.isfinite(state).all():
            raise ValueError("Invalid proprioceptive state")
        pair_mask = graph.node_mask[:, :, None] & graph.node_mask[:, None, :]
        pair_mask &= ~torch.eye(pair_mask.shape[1], dtype=torch.bool, device=state.device)[None]
        hidden = torch.zeros_like(pair_mask)
        input_graph = graph
        if self.training and mask_relations and self.config.graph.relation_loss_weight > 0:
            selected = (
                torch.rand(pair_mask.shape, device=state.device)
                < self.config.graph.relation_mask_probability
            )
            selected = torch.triu(selected, diagonal=1)
            hidden = (selected | selected.transpose(1, 2)) & pair_mask
            input_graph = graph.with_edges(graph.edges.masked_fill(hidden[..., None], 0))
        context = self.encode(vlm_inputs, input_graph)
        if noise is None:
            noise = torch.randn_like(actions)
        if noise.shape != actions.shape or not torch.isfinite(noise[valid]).all():
            raise ValueError("Invalid flow noise")
        if time is None:
            time = (
                torch.randn(b, device=state.device).sigmoid()
                if self.config.action.time_sampling == "logit_normal"
                else torch.rand(b, device=state.device)
            )
        if time.shape != (b,) or not ((time >= 0) & (time <= 1)).all():
            raise ValueError("time must be [B] in [0,1]")
        clean, noise = torch.where(valid, actions, 0), torch.where(valid, noise, 0)
        # Rectified flow convention: t=0 Gaussian noise, t=1 demonstration action.
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * clean
        prediction = self.action_expert(noisy, time, state, context, valid)
        flow_loss = (
            (prediction.float() - (clean - noise).float()).square() * valid
        ).sum() / valid.sum()
        relation_logits = self.graph_encoder.relation_logits(context.graph)
        relation_errors = F.binary_cross_entropy_with_logits(
            relation_logits.float(), graph.edges.float(), reduction="none"
        ).mean(-1)
        relation_loss = (relation_errors * hidden).sum() / hidden.sum().clamp_min(1)
        labeled = graph.target.ne(-100)
        target_loss = (
            F.cross_entropy(context.target_logits[labeled].float(), graph.target[labeled])
            if labeled.any()
            else context.graph.sum() * 0
        )
        loss = (
            flow_loss
            + self.config.graph.relation_loss_weight * relation_loss
            + self.config.graph.target_loss_weight * target_loss
        )
        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "relation_loss": relation_loss,
            "target_loss": target_loss,
            "target_logits": context.target_logits,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        vlm_inputs,
        graph,
        state,
        *,
        noise=None,
        action_mask=None,
        steps=None,
        solver=None,
        generator=None,
        ablate_graph=False,
    ):
        """Encode observations once, integrate a velocity field, return normalized actions.

        ablate_graph keeps the VLM input fixed and removes only the structured graph path.
        Useful for evaluation; it is not classifier-free guidance.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before sampling actions")
        cfg = self.config.action
        steps = cfg.inference_steps if steps is None else steps
        solver = cfg.solver if solver is None else solver
        if not isinstance(steps, int) or steps < 1 or solver not in {"euler", "heun"}:
            raise ValueError("Invalid integration steps or solver")
        if state.ndim != 2 or state.shape[1] != cfg.state_dim or not torch.isfinite(state).all():
            raise ValueError("Invalid proprioceptive state")
        b = state.shape[0]
        shape = (b, cfg.horizon, cfg.dim)
        x = (
            torch.randn(shape, device=state.device, generator=generator)
            if noise is None
            else noise.clone()
        )
        valid = self._valid_actions(x, action_mask)
        x = torch.where(valid, x, 0)
        context = self.encode(vlm_inputs, graph)
        if ablate_graph:
            context.graph = torch.zeros_like(context.graph[:, :1])
            context.graph_mask = torch.ones_like(context.graph_mask[:, :1])
        dt = 1.0 / steps
        for i in range(steps):
            time = torch.full((b,), i * dt, device=state.device)
            velocity = self.action_expert(x, time, state, context, valid)
            proposal = x + dt * velocity
            if solver == "heun":
                next_time = torch.full((b,), (i + 1) * dt, device=state.device)
                next_velocity = self.action_expert(proposal, next_time, state, context, valid)
                x = x + 0.5 * dt * (velocity + next_velocity)
            else:
                x = proposal
        return x.masked_fill(~valid, 0)

    def parameter_counts(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
