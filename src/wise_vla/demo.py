"""Generate a tiny synthetic dataset and optionally exercise training through inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import WISEConfig
from .data.dataset import EpisodeDataset
from .graph import WorldGraph


def make_demo_dataset(root, *, episodes=4, length=12, seed=42):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite demo dataset: {root}")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    entries = []
    for episode_index in range(episodes):
        directory = root / f"episode_{episode_index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        position = rng.uniform(0.1, 0.7, size=3)
        goal = position + np.array([0.1, 0.1, 0.05])
        states = np.zeros((length, 7), dtype=np.float32)
        states[:, :3] = np.linspace(position - 0.1, goal, length)
        states[:, 3:6] = rng.normal(0, 0.03, (length, 3))
        states[:, 6] = 1
        actions = np.zeros_like(states)
        actions[:, :3] = (goal - states[:, :3]) * 0.2
        actions[:, 3:6] = -states[:, 3:6] * 0.2
        actions[:, 6] = np.linspace(1, 0, length)
        np.save(directory / "states.npy", states, allow_pickle=False)
        np.save(directory / "actions.npy", actions, allow_pickle=False)
        rows = []
        for t in range(length):
            world = WorldGraph()
            world.upsert_node(
                "cube",
                label="red cube",
                node_type=1,
                position=position,
                size=(0.04, 0.04, 0.04),
                bbox=(0.15, 0.4, 0.35, 0.7),
                timestamp=t,
            )
            world.upsert_node(
                "tray",
                label="blue tray",
                node_type=3,
                position=goal,
                size=(0.2, 0.2, 0.03),
                bbox=(0.6, 0.4, 0.9, 0.8),
                timestamp=t,
            )
            world.set_relation("cube", "near", "tray", timestamp=t)
            image = Image.new("RGB", (96, 96), (225, 225, 225))
            draw = ImageDraw.Draw(image)
            draw.rectangle((16, 40, 36, 65), fill=(200, 40, 40))
            draw.rectangle((60, 40, 86, 75), fill=(45, 90, 190))
            draw.ellipse((int(10 + 3 * t), 20, int(18 + 3 * t), 28), fill=(40, 40, 40))
            filename = f"front_{t:04d}.png"
            image.save(directory / filename)
            rows.append(
                {
                    "instruction": "Move the red cube to the blue tray.",
                    "images": {"front": filename},
                    "graph": world.snapshot(t, target_id="cube"),
                }
            )
        (directory / "observations.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
        entries.append(
            {
                "id": directory.name,
                "path": directory.name,
                "split": "val" if episode_index == episodes - 1 else "train",
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "episodes": entries}, indent=2) + "\n")
    return manifest


def tiny_config():
    return WISEConfig(
        hidden_dim=64,
        num_heads=4,
        vlm={"backend": "tiny", "tiny_dim": 32, "dtype": "float32"},
        graph={"layers": 2, "max_nodes": 16},
        action={"layers": 2, "horizon": 8, "execute_steps": 2, "inference_steps": 3},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/demo")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.output)
    manifest = make_demo_dataset(root)
    print(f"Synthetic dataset: {manifest}")
    if args.train_steps:
        from .checkpoint import load_policy_model
        from .evaluate import evaluate
        from .policy import WISEPolicy
        from .train import TrainingConfig, run_training

        config = tiny_config()
        train = TrainingConfig(
            manifest=str(manifest),
            output_dir=str(root / "checkpoints"),
            batch_size=4,
            max_steps=args.train_steps,
            warmup_steps=2,
            learning_rate=0.001,
            ema_decay=0.9,
            log_every=5,
            checkpoint_every=10,
        )
        checkpoint = run_training(config, train)
        model, stats = load_policy_model(checkpoint)
        val = EpisodeDataset(manifest, config.action.horizon, split="val")
        result = evaluate(model, val, stats, batch_size=4)
        (root / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        sample = val[0]
        policy = WISEPolicy(model, stats)
        actions = policy.predict(
            **{key: sample[key] for key in ("images", "instruction", "state", "graph")}, seed=42
        )
        np.save(root / "predicted_actions.npy", actions, allow_pickle=False)
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "action_shape": list(actions.shape),
                    "evaluation": result,
                    "synthetic_only": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
