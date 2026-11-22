from types import SimpleNamespace

import numpy as np
import pytest

from wise_vla.data.dataset import EpisodeDataset
from wise_vla.graph_reach import MAX_STEP, MAX_STEPS, Layout, benchmark, generate
from wise_vla.policy import WISEPolicy


def test_paired_demos_and_split(tmp_path):
    manifest = generate(tmp_path / "data", train_pairs=1, val_pairs=1, seed=7)
    train = EpisodeDataset(manifest, horizon=4, split="train")
    left, right = train[0], train[train.episodes[0]["length"]]
    np.testing.assert_array_equal(left["state"], right["state"])
    np.testing.assert_array_equal(left["images"]["top"], right["images"]["top"])
    assert left["instruction"] == right["instruction"]
    assert left["graph"]["nodes"] == right["graph"]["nodes"]
    assert left["graph"]["target_id"] == "bin0"
    assert right["graph"]["target_id"] == "bin1"
    assert "target_id" not in Layout.sample(7, "train", 0).graph(0)
    assert {row["graph"]["edges"][0]["target"] for row in (left, right)} == {"bin0", "bin1"}
    assert left["actions"][0, 0] < 0 < right["actions"][0, 0]
    val = EpisodeDataset(manifest, horizon=4, split="val")
    assert not np.array_equal(left["state"], val[0]["state"])
    assert not np.array_equal(val[0]["state"], Layout.sample(7, "test", 0).start)
    with pytest.raises(FileExistsError):
        generate(manifest.parent)


class Oracle:
    """Follow the supplied relation. Choose the first bin when the relation is absent."""

    model = SimpleNamespace(config=SimpleNamespace(action=SimpleNamespace(dim=2, state_dim=2)))

    def reset(self):
        self.seeds = []

    def act(self, *, images, instruction, state, graph, seed):
        self.seeds.append(seed)
        target = graph["edges"][0]["target"] if graph["edges"] else "bin0"
        node = next(n for n in graph["nodes"] if n["id"] == target)
        return node["features"][:2] - state


def test_rollouts_score_both_goals_and_match_noise():
    class RecordingOracle(Oracle):
        def reset(self):
            super().reset()
            histories.append(self.seeds)

    histories = []
    result = benchmark(RecordingOracle(), pairs=3)
    assert result["metrics"]["full"] == {
        "episodes": 6,
        "successes": 6,
        "success_rate": 1.0,
        "both_goals_reached_rate": 1.0,
    }
    assert result["metrics"]["no_edges"]["success_rate"] == 0.5
    assert result["metrics"]["no_edges"]["both_goals_reached_rate"] == 0
    assert [r["outcome"] for r in result["runs"][:4]] == [
        "success",
        "success",
        "success",
        "wrong_bin",
    ]
    for start in range(0, len(histories), 4):
        # Rollouts can terminate at different times. Their common prefix must match.
        assert all(len(set(group)) == 1 for group in zip(*histories[start : start + 4]))


def test_timeout_and_displacement_limit():
    class Stationary(Oracle):
        def act(self, **kwargs):
            return [0, 0]

    runs = benchmark(Stationary(), pairs=1)["runs"]
    assert all(r["outcome"] == "timeout" and r["steps"] == MAX_STEPS for r in runs)
    layout = Layout.sample(42, "test", 3)
    after, _ = layout.step(layout.start, [1, -1], 0)
    assert np.linalg.norm(after - layout.start) == pytest.approx(MAX_STEP)
    with pytest.raises(ValueError):
        layout.step(layout.start, [np.nan, 0], 1)


def test_checkpoint_rollout_is_repeatable(tmp_path, config):
    from wise_vla.checkpoint import save_checkpoint
    from wise_vla.data.normalization import NormalizationStats
    from wise_vla.model import WISEVLA

    config.action.state_dim = config.action.dim = 2
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, WISEVLA(config), stats=NormalizationStats([0, 0], [1, 1], [0, 0], [0.05, 0.05])
    )
    policy = WISEPolicy.from_checkpoint(path)
    a = benchmark(policy, pairs=1)
    assert a == benchmark(policy, pairs=1)
    assert len(a["runs"]) == 4
