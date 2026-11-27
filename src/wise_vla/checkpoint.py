"""Atomic resumable checkpoints; frozen pretrained weights are referenced, not copied."""

import hashlib
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from .config import WISEConfig
from .data.normalization import NormalizationStats
from .model import WISEVLA


class EMA:
    def __init__(self, model, decay=0.999):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = decay
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1 - self.decay)

    @contextmanager
    def apply(self, model):
        parameters = dict(model.named_parameters())
        original = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in original.items():
                    parameters[name].copy_(value)


def save_checkpoint(
    path,
    model,
    *,
    stats,
    optimizer=None,
    scheduler=None,
    ema=None,
    trainer_state=None,
    rng_states=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    skip_backbone = model.config.vlm.frozen and model.config.vlm.backend != "tiny"
    model_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not (skip_backbone and name.startswith("backbone."))
    }
    payload = {
        "format_version": 1,
        "config": model.config.to_dict(),
        "model": model_state,
        "excluded_frozen_backbone": skip_backbone,
        "normalization": asdict(stats),
        "torch_version": str(torch.__version__),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "ema": None if ema is None else {k: v.detach().cpu() for k, v in ema.shadow.items()},
        "ema_decay": None if ema is None else ema.decay,
        "trainer": trainer_state or {},
        "rng_states": rng_states,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    return payload


def checkpoint_sha256(stream):
    """Fingerprint the same open file that will be loaded, even if latest.pt is replaced."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def check_model_config(config, payload, *, new_stage=False):
    saved = WISEConfig(**payload["config"])
    if new_stage:
        # A new dataset may have different graph supervision coverage.
        saved.graph.relation_loss_weight = config.graph.relation_loss_weight
        saved.graph.target_loss_weight = config.graph.target_loss_weight
        saved.graph.relation_mask_probability = config.graph.relation_mask_probability
    if config != saved:
        raise ValueError("Checkpoint/model configuration mismatch")


def restore_weights(model, payload, *, use_ema=False, new_stage=False):
    check_model_config(model.config, payload, new_stage=new_stage)
    incompat = model.load_state_dict(payload["model"], strict=False)
    allowed_missing = {name for name in model.state_dict() if name.startswith("backbone.")}
    missing = set(incompat.missing_keys)
    if incompat.unexpected_keys or (
        missing and not (payload["excluded_frozen_backbone"] and missing == allowed_missing)
    ):
        raise ValueError(f"Incompatible checkpoint keys: {incompat}")
    if use_ema and payload["ema"] is not None:
        parameters = dict(model.named_parameters())
        expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
        if set(payload["ema"]) != expected:
            raise ValueError("EMA parameter set mismatch")
        with torch.no_grad():
            for name, value in payload["ema"].items():
                parameters[name].copy_(value)


def load_policy_model(path, device="cpu", *, use_ema=True):
    payload = read_checkpoint(path)
    model = WISEVLA(WISEConfig(**payload["config"])).to(device)
    restore_weights(model, payload, use_ema=use_ema)
    return model.eval(), NormalizationStats(**payload["normalization"])
