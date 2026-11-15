"""Robot-facing interface. Returns actions; never commands hardware directly."""

import hashlib
import json
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .backbones import ObservationProcessor
from .checkpoint import load_policy_model
from .graph import GraphTensorizer


@dataclass
class ActionBounds:
    """Optional bounds in the robot's denormalized action units; reject, don't silently clip."""

    low: list[float]
    high: list[float]

    def validate(self, actions):
        low, high = np.asarray(self.low), np.asarray(self.high)
        if low.shape != (actions.shape[-1],) or high.shape != low.shape:
            raise ValueError("Action bound dimensions do not match the robot")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or (low > high).any():
            raise ValueError("Invalid action bounds")
        if (actions < low).any() or (actions > high).any():
            raise ValueError("Predicted action chunk violates configured action bounds")


class WISEPolicy:
    def __init__(self, model, stats, *, bounds: ActionBounds | None = None):
        self.model, self.stats, self.bounds = model.eval(), stats, bounds
        self.device = next(model.parameters()).device
        self.processor = ObservationProcessor(model.config)
        self.tensorizer = GraphTensorizer(model.config.graph)
        self.reset()

    @classmethod
    def from_checkpoint(cls, path, device="cpu", *, bounds=None, use_ema=True):
        model, stats = load_policy_model(path, device, use_ema=use_ema)
        return cls(model, stats, bounds=bounds)

    def reset(self):
        self.queue = deque()
        self.last_context = None

    @torch.inference_mode()
    def predict(self, *, images, instruction, state, graph, seed=None):
        batch_graph = self.tensorizer([graph])
        inputs = self.processor([{"images": images, "instruction": instruction}], batch_graph)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        normalized = self.stats.normalize_state(state)
        generator = (
            torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        )
        actions = self.model.sample_actions(
            inputs, batch_graph.to(self.device), normalized, generator=generator
        )
        actions = self.stats.denormalize_action(actions)[0].cpu().numpy()
        if not np.isfinite(actions).all():
            raise FloatingPointError("Policy generated non-finite actions")
        if self.bounds:
            self.bounds.validate(actions)
        return actions

    def act(self, *, images, instruction, state, graph, force_replan=False, seed=None):
        # Invalidate cached actions on any graph change, including geometry or stale-node removal.
        signature = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
        context = (instruction, signature)
        if force_replan or context != self.last_context:
            self.queue.clear()
        if not self.queue:
            chunk = self.predict(
                images=images, instruction=instruction, state=state, graph=graph, seed=seed
            )
            self.queue.extend(chunk[: self.model.config.action.execute_steps])
            self.last_context = context
        return self.queue.popleft().copy()
