"""Streaming training-split statistics; stored with every checkpoint."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class NormalizationStats:
    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]

    def __post_init__(self):
        for kind in ("state", "action"):
            mean, std = (
                np.array(getattr(self, f"{kind}_mean")),
                np.array(getattr(self, f"{kind}_std")),
            )
            if mean.ndim != 1 or len(mean) == 0 or mean.shape != std.shape:
                raise ValueError("Invalid normalization dimensions")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Normalization statistics must be finite with positive std")

    def transform(self, x, kind, inverse=False):
        mean = torch.as_tensor(getattr(self, f"{kind}_mean"), device=x.device, dtype=x.dtype)
        std = torch.as_tensor(getattr(self, f"{kind}_std"), device=x.device, dtype=x.dtype)
        if x.shape[-1] != mean.numel():
            raise ValueError("Normalization dimensions do not match input")
        return x * std + mean if inverse else (x - mean) / std

    def normalize_state(self, x):
        return self.transform(x, "state")

    def normalize_action(self, x):
        return self.transform(x, "action")

    def denormalize_action(self, x):
        return self.transform(x, "action", inverse=True)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class RunningMoments:
    def __init__(self, dim):
        self.count = np.zeros(dim, dtype=np.float64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, values, valid=None):
        values = np.asarray(values, dtype=np.float64)
        valid = np.ones_like(values, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if not np.isfinite(values[valid]).all():
            raise ValueError("Non-finite values in statistics input")
        n = valid.sum(0)
        values = np.where(valid, values, 0)
        mean = values.sum(0) / np.maximum(n, 1)
        m2 = (np.where(valid, values - mean, 0) ** 2).sum(0)
        total = self.count + n
        delta = mean - self.mean
        self.m2 += m2 + delta**2 * self.count * n / np.maximum(total, 1)
        self.mean += delta * n / np.maximum(total, 1)
        self.count = total

    def finish(self):
        if (self.count == 0).any():
            raise ValueError("Every dimension requires at least one valid training observation")
        return self.mean.tolist(), np.sqrt(self.m2 / self.count).clip(1e-3).tolist()


def fit_stats(dataset):
    state_stats = RunningMoments(dataset.episodes[0]["state_dim"])
    action_stats = RunningMoments(dataset.episodes[0]["action_dim"])
    for index, episode in enumerate(dataset.episodes):
        states, actions = dataset.arrays(index)
        # Count each timestep once, not once per overlapping action chunk.
        for start in range(0, len(actions), 8192):
            end = min(start + 8192, len(actions))
            mask = np.array(
                [
                    obs.get("action_dim_mask", [True] * actions.shape[1])
                    for obs in episode["observations"][start:end]
                ],
                dtype=bool,
            )
            state_stats.update(states[start:end])
            action_stats.update(actions[start:end], mask)
    state_mean, state_std = state_stats.finish()
    action_mean, action_std = action_stats.finish()
    return NormalizationStats(state_mean, state_std, action_mean, action_std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="normalization.json")
    args = parser.parse_args()
    from .dataset import EpisodeDataset

    stats = fit_stats(EpisodeDataset(args.manifest, 1, split=args.split))
    stats.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
