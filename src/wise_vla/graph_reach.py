"""Paired closed-loop reaching task for testing the use of a directed graph relation."""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .graph import WorldGraph
from .policy import WISEPolicy

INSTRUCTION = "Reach the closed bin containing the part."
MAX_STEP = 0.08
RADIUS = 0.065
MAX_STEPS = 16
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


@dataclass
class Layout:
    bins: np.ndarray
    start: np.ndarray

    @classmethod
    def sample(cls, seed, split, index):
        rng = np.random.default_rng([seed, SPLIT_IDS[split], index])
        bins = rng.uniform([0.14, 0.45], [0.26, 0.65], size=(2, 2))
        bins[1, 0] = 1 - bins[1, 0]
        return cls(bins, rng.uniform([0.45, 0.18], [0.55, 0.32]))

    def graph(self, goal):
        world = WorldGraph()
        # The part is hidden. Its node uses the same placeholder geometry in both episodes.
        world.upsert_node("part", label="part", node_type=1, position=(0, 0, 0), timestamp=0)
        for i, (x, y) in enumerate(self.bins):
            world.upsert_node(
                f"bin{i}",
                label="closed bin",
                node_type=3,
                position=(float(x), float(y), 0),
                timestamp=0,
            )
        world.set_relation("part", "inside", f"bin{goal}", timestamp=0)
        return world.snapshot(0)

    def render(self, position):
        image = Image.new("RGB", (64, 64), (224, 222, 216))
        draw = ImageDraw.Draw(image)
        for x, y in self.bins * 63:
            draw.rectangle((x - 3, y - 3, x + 3, y + 3), fill=(54, 82, 109))
        x, y = position * 63
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(192, 79, 48))
        return image

    @staticmethod
    def displacement(action):
        delta = np.asarray(action, dtype=np.float64)
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise ValueError("Expected a finite two-dimensional displacement")
        return delta * (MAX_STEP / max(MAX_STEP, float(np.linalg.norm(delta))))

    def step(self, position, action, goal):
        position = np.clip(position + self.displacement(action), 0, 1)
        distances = np.linalg.norm(self.bins - position, axis=1)
        nearest = int(np.argmin(distances))
        outcome = None
        if distances[nearest] <= RADIUS:
            outcome = "success" if nearest == goal else "wrong_bin"
        return position, outcome


def _demonstration(directory, layout, goal):
    directory.mkdir()
    states, actions = [], []
    position = layout.start
    graph = layout.graph(goal)
    # The existing target loss reads this label. Online observations omit it.
    graph["target_id"] = f"bin{goal}"
    (directory / "graph.json").write_text(json.dumps(graph))
    with (directory / "observations.jsonl").open("w") as rows:
        for index in range(MAX_STEPS):
            direction = layout.bins[goal] - position
            action = layout.displacement(direction)
            states.append(position)
            actions.append(action)
            filename = f"{index:03d}.png"
            layout.render(position).save(directory / filename)
            rows.write(
                json.dumps(
                    {"instruction": INSTRUCTION, "images": {"top": filename}, "graph": "graph.json"}
                )
                + "\n"
            )
            position, outcome = layout.step(position, action, goal)
            if outcome == "success":
                break
        else:
            raise RuntimeError("Expert did not reach the goal")
    np.save(directory / "states.npy", np.asarray(states, dtype=np.float32))
    np.save(directory / "actions.npy", np.asarray(actions, dtype=np.float32))


def generate(output, *, train_pairs=32, val_pairs=8, seed=42):
    """Keep both counterfactual episodes of a layout in the same split."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if min(train_pairs, val_pairs) < 1 or seed < 0:
        raise ValueError("Use positive pair counts and a nonnegative seed")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "dataset"
        staged.mkdir()
        episodes = []
        for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
            for i in range(pairs):
                layout = Layout.sample(seed, split, i)
                for goal in (0, 1):
                    name = f"{split}_{i:05d}_{goal}"
                    _demonstration(staged / name, layout, goal)
                    episodes.append({"id": name, "path": name, "split": split})
        manifest = {"schema_version": 1, "task": "graph_reach", "seed": seed, "episodes": episodes}
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)
    return output / "manifest.json"


def _rollout(policy, layout, goal, graph, noise_seeds):
    policy.reset()
    position = layout.start
    for i, seed in enumerate(noise_seeds, 1):
        action = policy.act(
            images={"top": layout.render(position)},
            instruction=INSTRUCTION,
            state=position,
            graph=graph,
            seed=int(seed),
        )
        position, outcome = layout.step(position, action, goal)
        if outcome:
            return outcome, i
    return "timeout", len(noise_seeds)


def benchmark(policy, *, pairs=16, seed=42):
    if pairs < 1 or seed < 0:
        raise ValueError("Use a positive pair count and a nonnegative seed")
    cfg = policy.model.config.action
    if (cfg.state_dim, cfg.dim) != (2, 2):
        raise ValueError("Graph reach requires a checkpoint with 2D state and displacement")
    runs = []
    for i in range(pairs):
        layout = Layout.sample(seed, "test", i)
        # All four rollouts start identically and draw the same noise at a given step.
        noise_seeds = np.random.default_rng([seed, 3, i]).integers(2**31, size=MAX_STEPS)
        for goal in (0, 1):
            graph = layout.graph(goal)
            for condition, inputs in [("full", graph), ("no_edges", dict(graph, edges=[]))]:
                outcome, steps = _rollout(policy, layout, goal, inputs, noise_seeds)
                runs.append(
                    {
                        "pair": i,
                        "goal": goal,
                        "condition": condition,
                        "outcome": outcome,
                        "steps": steps,
                    }
                )
    metrics = {}
    for condition in ("full", "no_edges"):
        trials = [r for r in runs if r["condition"] == condition]
        successes = [r["pair"] for r in trials if r["outcome"] == "success"]
        both = sum(count == 2 for count in Counter(successes).values())
        metrics[condition] = {
            "successes": len(successes),
            "episodes": len(trials),
            "success_rate": len(successes) / len(trials),
            "both_goals_reached_rate": both / pairs,
        }
    return {"task": "graph_reach", "seed": seed, "pairs": pairs, "metrics": metrics, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("generate", help="Write expert demonstrations in WISE format")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--train-pairs", type=int, default=32)
    data.add_argument("--val-pairs", type=int, default=8)
    run = commands.add_parser("evaluate", help="Run paired rollouts on held-out layouts")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--pairs", type=int, default=16)
    run.add_argument("--device", default="cpu")
    run.add_argument("--output", type=Path)
    for sub in (data, run):
        sub.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "generate":
        print(
            generate(
                args.output, train_pairs=args.train_pairs, val_pairs=args.val_pairs, seed=args.seed
            )
        )
    else:
        policy = WISEPolicy.from_checkpoint(args.checkpoint, device=args.device)
        result = benchmark(policy, pairs=args.pairs, seed=args.seed)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
