"""Strict, serializable configuration; unknown keys are errors."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class VLMConfig:
    backend: str = "qwen3_vl"
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    revision: str = "main"
    frozen: bool = True
    dtype: str = "bfloat16"
    tiny_dim: int = 64
    max_text_length: int = 256
    max_pixels: int = 262144
    local_files_only: bool = False


@dataclass
class GraphConfig:
    node_feature_dim: int = 16
    num_node_types: int = 8
    relations: list[str] = field(
        default_factory=lambda: [
            "on",
            "inside",
            "near",
            "holding",
            "reachable",
            "part_of",
            "target",
            "blocked_by",
        ]
    )
    layers: int = 3
    max_nodes: int = 64
    max_label_tokens: int = 32
    relation_loss_weight: float = 0.1
    target_loss_weight: float = 0.1
    relation_mask_probability: float = 0.15


@dataclass
class ActionConfig:
    dim: int = 7
    state_dim: int = 7
    horizon: int = 16
    execute_steps: int = 4
    layers: int = 6
    inference_steps: int = 10
    solver: str = "heun"
    time_sampling: str = "logit_normal"
    condition_on_target: bool = False


@dataclass
class WISEConfig:
    hidden_dim: int = 384
    num_heads: int = 6
    dropout: float = 0.0
    vlm: VLMConfig = field(default_factory=VLMConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    def __post_init__(self):
        for name, cls in (("vlm", VLMConfig), ("graph", GraphConfig), ("action", ActionConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))
        positive = [
            self.hidden_dim,
            self.num_heads,
            self.vlm.tiny_dim,
            self.graph.node_feature_dim,
            self.graph.num_node_types,
            self.graph.max_nodes,
            self.graph.max_label_tokens,
            self.graph.layers,
            self.action.dim,
            self.action.state_dim,
            self.action.horizon,
            self.action.layers,
            self.action.inference_steps,
            self.vlm.max_pixels,
            self.vlm.max_text_length,
        ]
        if any(x <= 0 for x in positive):
            raise ValueError("Dimensions, depths and sequence limits must be positive")
        if self.hidden_dim % self.num_heads or (self.hidden_dim // self.num_heads) % 2:
            raise ValueError("hidden_dim must be divisible by num_heads with even head dimension")
        if not 1 <= self.action.execute_steps <= self.action.horizon:
            raise ValueError("execute_steps must lie in [1, horizon]")
        if self.vlm.backend not in {"tiny", "qwen3_vl"}:
            raise ValueError("Supported VLM backends: tiny, qwen3_vl")
        if self.vlm.dtype not in {"float32", "bfloat16"}:
            raise ValueError("VLM dtype must be float32 or bfloat16")
        if self.action.solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")
        if self.action.time_sampling not in {"uniform", "logit_normal"}:
            raise ValueError("time_sampling must be uniform or logit_normal")
        if not isinstance(self.action.condition_on_target, bool):
            raise ValueError("condition_on_target must be a boolean")
        if not 0 <= self.dropout < 1 or not 0 <= self.graph.relation_mask_probability <= 1:
            raise ValueError("Invalid dropout or relation mask probability")
        if min(self.graph.relation_loss_weight, self.graph.target_loss_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if (
            not self.graph.relations
            or any(not isinstance(r, str) or not r for r in self.graph.relations)
            or len(set(self.graph.relations)) != len(self.graph.relations)
        ):
            raise ValueError("Relation names must be nonempty and unique")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        with open(path) as stream:
            values = yaml.safe_load(stream)
        return cls(**values.get("model", values))

    def save(self, path: str | Path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
