import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..backbones import ObservationProcessor
from ..graph import GraphTensorizer


def safe_path(root: Path, relative: str) -> Path:
    """Datasets are portable and may not escape their declared root."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


class EpisodeDataset(Dataset):
    """Memory-mapped arrays + one JSON observation per timestep.

    Splitting happens at episode granularity. Action chunks are always bounded by
    the current episode and padded with an explicit mask.
    """

    def __init__(self, manifest, horizon, *, split=None):
        self.manifest = Path(manifest).resolve()
        content = json.loads(self.manifest.read_text())
        if content.get("schema_version") != 1:
            raise ValueError("Expected dataset schema_version=1")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon, self.episodes, self.index = horizon, [], []
        seen = set()
        for entry in content["episodes"]:
            if entry["id"] in seen:
                raise ValueError("Duplicate episode ID in manifest")
            seen.add(entry["id"])
            if split is not None and entry.get("split") != split:
                continue
            path = safe_path(self.manifest.parent, entry["path"])
            states = np.load(path / "states.npy", mmap_mode="r", allow_pickle=False)
            actions = np.load(path / "actions.npy", mmap_mode="r", allow_pickle=False)
            if (
                states.ndim != 2
                or actions.ndim != 2
                or len(states) != len(actions)
                or len(actions) == 0
            ):
                raise ValueError(f"Invalid episode arrays: {path}")
            observations = [
                json.loads(line)
                for line in (path / "observations.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if len(observations) != len(actions):
                raise ValueError(f"Observation/action length mismatch: {path}")
            episode_index = len(self.episodes)
            self.episodes.append(
                {
                    "path": path,
                    "observations": observations,
                    "length": len(actions),
                    "id": entry["id"],
                    "action_dim": actions.shape[1],
                    "state_dim": states.shape[1],
                }
            )
            self.index.extend((episode_index, t) for t in range(len(actions)))
        if not self.episodes:
            raise ValueError(f"No episodes for split {split!r}")

    def __len__(self):
        return len(self.index)

    @lru_cache(maxsize=8)
    def arrays(self, episode_index):
        path = self.episodes[episode_index]["path"]
        return (
            np.load(path / "states.npy", mmap_mode="r", allow_pickle=False),
            np.load(path / "actions.npy", mmap_mode="r", allow_pickle=False),
        )

    def __getitem__(self, index):
        episode_index, t = self.index[index]
        episode = self.episodes[episode_index]
        states, actions = self.arrays(episode_index)
        obs = episode["observations"][t]
        chunk = np.zeros((self.horizon, actions.shape[1]), dtype=np.float32)
        count = min(self.horizon, len(actions) - t)
        chunk[:count] = actions[t : t + count]
        action_mask = np.zeros_like(chunk, dtype=bool)
        action_mask[:count] = True
        for future in range(count):
            future_obs = episode["observations"][t + future]
            if "action_dim_mask" in future_obs:
                dim_mask = np.asarray(future_obs["action_dim_mask"], dtype=bool)
                if dim_mask.shape != (actions.shape[1],):
                    raise ValueError("action_dim_mask must have one entry per action dimension")
                action_mask[future] &= dim_mask
        images = {}
        for camera, relative in obs["images"].items():
            with Image.open(safe_path(episode["path"], relative)) as image:
                images[camera] = image.convert("RGB")
        graph = obs.get("graph", {"nodes": [], "edges": []})
        if isinstance(graph, str):
            graph = json.loads(safe_path(episode["path"], graph).read_text())
        return {
            "instruction": obs["instruction"],
            "images": images,
            "graph": graph,
            "state": np.array(states[t], dtype=np.float32),
            "actions": chunk,
            "action_mask": action_mask,
            "episode_id": episode["id"],
            "timestep": t,
        }


class PolicyCollator:
    def __init__(self, config, stats=None):
        self.config, self.stats = config, stats
        self.graphs = GraphTensorizer(config.graph)
        self.processor = ObservationProcessor(config)

    def __call__(self, samples):
        graph = self.graphs([sample["graph"] for sample in samples])
        state = torch.tensor(np.stack([s["state"] for s in samples]), dtype=torch.float32)
        actions = torch.tensor(np.stack([s["actions"] for s in samples]), dtype=torch.float32)
        mask = torch.tensor(np.stack([s["action_mask"] for s in samples]), dtype=torch.bool)
        if (
            state.shape[-1] != self.config.action.state_dim
            or actions.shape[-1] != self.config.action.dim
        ):
            raise ValueError("Dataset dimensions do not match the policy configuration")
        if not torch.isfinite(state).all() or not torch.isfinite(actions[mask]).all():
            raise ValueError("Dataset contains non-finite state/action values")
        if self.stats:
            state = self.stats.normalize_state(state)
            actions = self.stats.normalize_action(actions)
        actions = torch.where(mask, actions, 0)
        return {
            "vlm_inputs": self.processor(samples, graph),
            "graph": graph,
            "state": state,
            "actions": actions,
            "action_mask": mask,
        }


def batch_to(batch, device):
    return {
        key: (
            {k: v.to(device) for k, v in value.items()}
            if isinstance(value, dict)
            else value.to(device)
        )
        for key, value in batch.items()
    }
