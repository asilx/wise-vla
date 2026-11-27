"""Paired closed-loop reaching task for testing the use of a directed graph relation."""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from .checkpoint import checkpoint_sha256
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


def _expert_trajectory(layout, goal, position):
    trajectory = []
    for _ in range(MAX_STEPS):
        action = layout.displacement(layout.bins[goal] - position)
        trajectory.append((position, action))
        position, outcome = layout.step(position, action, goal)
        if outcome:
            return trajectory if outcome == "success" else []
    return []


def _write_demonstration(directory, layout, goal, trajectory):
    directory.mkdir()
    graph = layout.graph(goal)
    # The existing target loss reads this label. Online observations omit it.
    graph["target_id"] = f"bin{goal}"
    (directory / "graph.json").write_text(json.dumps(graph))
    with (directory / "observations.jsonl").open("w") as rows:
        for index, (position, _) in enumerate(trajectory):
            filename = f"{index:03d}.png"
            layout.render(position).save(directory / filename)
            rows.write(
                json.dumps(
                    {"instruction": INSTRUCTION, "images": {"top": filename}, "graph": "graph.json"}
                )
                + "\n"
            )
    states, actions = zip(*trajectory)
    np.save(directory / "states.npy", np.asarray(states, dtype=np.float32))
    np.save(directory / "actions.npy", np.asarray(actions, dtype=np.float32))


def _collect_correction(staged, name, layout, goal, policy, noise_seeds):
    visited = []
    outcome, _ = _rollout(policy, layout, goal, layout.graph(goal), noise_seeds, visited=visited)
    # Project each learner state onto the finite nominal path, including its endpoints.
    direction = layout.bins[goal] - layout.start
    offsets = np.asarray(visited) - layout.start
    progress = np.clip((offsets @ direction) / (direction @ direction), 0, 1)
    deviation = np.linalg.norm(offsets - progress[:, None] * direction, axis=1)
    for step in np.argsort(-deviation, kind="stable"):
        if deviation[step] <= 1e-6:
            break
        trajectory = _expert_trajectory(layout, goal, visited[step])
        if trajectory:
            episode = name + "_recovery"
            _write_demonstration(staged / episode, layout, goal, trajectory)
            return {
                "id": episode,
                "path": episode,
                "split": "train",
                "kind": "recovery",
                "nominal_episode": name,
                "takeover_step": int(step),
                "learner_outcome": outcome,
            }
    return None


def generate(output, *, train_pairs=32, val_pairs=8, seed=42, policy=None, source_sha256=None):
    """Write paired demonstrations and optionally add expert recoveries on training layouts."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if min(train_pairs, val_pairs) < 1 or seed < 0:
        raise ValueError("Use positive pair counts and a nonnegative seed")
    if policy is not None:
        _check_policy_dimensions(policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "dataset"
        staged.mkdir()
        episodes = []
        for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
            for i in range(pairs):
                layout = Layout.sample(seed, split, i)
                collect = policy is not None and split == "train"
                if collect:
                    noise_seeds = np.random.default_rng([seed, 4, i]).integers(
                        2**31, size=MAX_STEPS
                    )
                for goal in (0, 1):
                    name = f"{split}_{i:05d}_{goal}"
                    nominal = _expert_trajectory(layout, goal, layout.start)
                    if not nominal:
                        raise RuntimeError("Expert did not reach the goal from the nominal start")
                    _write_demonstration(staged / name, layout, goal, nominal)
                    entry = {"id": name, "path": name, "split": split}
                    if policy is not None:
                        entry["kind"] = "nominal"
                    episodes.append(entry)
                    if collect:
                        recovery = _collect_correction(
                            staged, name, layout, goal, policy, noise_seeds
                        )
                        if recovery is not None:
                            episodes.append(recovery)
        manifest = {"schema_version": 1, "task": "graph_reach", "seed": seed, "episodes": episodes}
        if policy is not None:
            manifest["collection"] = {
                "checkpoint_sha256": source_sha256,
                "training_rollouts": 2 * train_pairs,
                "recovery_episodes": sum(e["kind"] == "recovery" for e in episodes),
            }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staged.rename(output)
    return output / "manifest.json"


def _rollout(policy, layout, goal, graph, noise_seeds, *, visited=None):
    policy.reset()
    position = layout.start
    for i, seed in enumerate(noise_seeds, 1):
        if visited is not None:
            visited.append(position.copy())
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


def _check_policy_dimensions(policy):
    cfg = policy.model.config.action
    if (cfg.state_dim, cfg.dim) != (2, 2):
        raise ValueError("Graph reach requires a checkpoint with 2D state and displacement")


def benchmark(policy, *, pairs=16, seed=42):
    if pairs < 1 or seed < 0:
        raise ValueError("Use a positive pair count and a nonnegative seed")
    _check_policy_dimensions(policy)
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
    collect = commands.add_parser(
        "collect-corrections", help="Add expert recoveries from learner rollouts"
    )
    for sub in (data, collect):
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--train-pairs", type=int, default=32)
        sub.add_argument("--val-pairs", type=int, default=8)
    run = commands.add_parser("evaluate", help="Run paired rollouts on held-out layouts")
    for sub in (collect, run):
        sub.add_argument("--checkpoint", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
    run.add_argument("--pairs", type=int, default=16)
    run.add_argument("--output", type=Path)
    for sub in (data, collect, run):
        sub.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command in {"generate", "collect-corrections"}:
        policy, digest = None, None
        if args.command == "collect-corrections":
            with args.checkpoint.open("rb") as stream:
                digest = checkpoint_sha256(stream)
                policy = WISEPolicy.from_checkpoint(stream, device=args.device)
        manifest = generate(
            args.output,
            train_pairs=args.train_pairs,
            val_pairs=args.val_pairs,
            seed=args.seed,
            policy=policy,
            source_sha256=digest,
        )
        if policy is not None:
            print(json.dumps(json.loads(manifest.read_text())["collection"]))
        print(manifest)
    else:
        policy = WISEPolicy.from_checkpoint(args.checkpoint, device=args.device)
        result = benchmark(policy, pairs=args.pairs, seed=args.seed)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
