import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from wise_vla.checkpoint import EMA, load_policy_model, read_checkpoint, save_checkpoint
from wise_vla.data.dataset import EpisodeDataset, PolicyCollator, safe_path
from wise_vla.data.normalization import NormalizationStats, fit_stats
from wise_vla.demo import make_demo_dataset
from wise_vla.evaluate import evaluate
from wise_vla.policy import ActionBounds, WISEPolicy
from wise_vla.train import TrainingConfig, run_training


@pytest.fixture
def dataset(tmp_path, config):
    manifest = make_demo_dataset(tmp_path / "dataset", episodes=3, length=5)
    return EpisodeDataset(manifest, config.action.horizon, split="train")


def test_episode_boundaries_and_split(dataset):
    last = dataset[4]
    assert last["action_mask"].sum() == 7
    assert np.count_nonzero(last["actions"][1:]) == 0
    assert dataset[5]["timestep"] == 0
    val = EpisodeDataset(dataset.manifest, 8, split="val")
    assert {e["id"] for e in val.episodes}.isdisjoint({e["id"] for e in dataset.episodes})


def test_action_dimension_mask_follows_future_timestep(dataset):
    dataset.episodes[0]["observations"][2]["action_dim_mask"] = [True] * 6 + [False]
    sample = dataset[0]
    assert sample["action_mask"][1, 6]
    assert not sample["action_mask"][2, 6]


def test_statistics_count_timesteps_once(dataset):
    stats = fit_stats(dataset)
    all_actions = np.concatenate([dataset.arrays(i)[1] for i in range(len(dataset.episodes))])
    np.testing.assert_allclose(stats.action_mean, all_actions.mean(0), atol=1e-6)
    np.testing.assert_allclose(stats.action_std, all_actions.std(0).clip(1e-3), atol=1e-6)
    actions = torch.randn(3, 8, 7)
    torch.testing.assert_close(stats.denormalize_action(stats.normalize_action(actions)), actions)


def test_dataset_root_and_invalid_stats(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../outside")
    with pytest.raises(ValueError):
        NormalizationStats([0], [0], [0], [1])


def test_checkpoint_roundtrip_preserves_frozen_tiny(model, batch, tmp_path):
    stats = NormalizationStats([0.0] * 7, [1.0] * 7, [0.0] * 7, [1.0] * 7)
    model.eval()
    noise = torch.randn_like(batch["actions"])
    inputs = {k: batch[k] for k in ("vlm_inputs", "graph", "state")}
    before = model.sample_actions(**inputs, noise=noise)
    path = tmp_path / "policy.pt"
    save_checkpoint(path, model, stats=stats)
    torch.manual_seed(999)
    restored, restored_stats = load_policy_model(path)
    after = restored.sample_actions(**inputs, noise=noise)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert restored_stats == stats
    assert not read_checkpoint(path)["excluded_frozen_backbone"]


def test_ema_context_restores_even_on_error(model):
    ema = EMA(model, decay=0.5)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(2)
    ema.update(model)
    with pytest.raises(RuntimeError):
        with ema.apply(model):
            torch.testing.assert_close(parameter, original + 1)
            raise RuntimeError("intentional")
    torch.testing.assert_close(parameter, original + 2)


def test_training_resume_matches_uninterrupted(config, dataset, tmp_path):
    settings = TrainingConfig(
        manifest=str(dataset.manifest),
        output_dir=str(tmp_path / "full"),
        max_steps=6,
        batch_size=3,
        grad_accumulation=2,
        warmup_steps=2,
        checkpoint_every=2,
        log_every=10,
        learning_rate=0.001,
    )
    full = run_training(copy.deepcopy(config), settings)
    partial_settings = replace(settings, output_dir=str(tmp_path / "resumed"))
    partial = run_training(copy.deepcopy(config), partial_settings, stop_after=3)
    resumed = run_training(copy.deepcopy(config), partial_settings, resume=partial)
    a, b = read_checkpoint(full), read_checkpoint(resumed)
    assert a["trainer"]["step"] == b["trainer"]["step"] == 6
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0, msg=key)
    for key in a["ema"]:
        torch.testing.assert_close(a["ema"][key], b["ema"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="settings"):
        run_training(copy.deepcopy(config), replace(partial_settings, batch_size=2), resume=resumed)


def test_tiny_model_learns_fixed_flow_batch(model, batch):
    noise, time = torch.randn_like(batch["actions"]), torch.full((2,), 0.5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003)
    initial = float(
        model(**batch, noise=noise, time=time, mask_relations=False)["flow_loss"].detach()
    )
    for _ in range(35):
        loss = model(**batch, noise=noise, time=time, mask_relations=False)["flow_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = float(
        model(**batch, noise=noise, time=time, mask_relations=False)["flow_loss"].detach()
    )
    assert final < initial * 0.2, (initial, final)


def test_policy_and_evaluation(config, model, dataset, tmp_path):
    stats = fit_stats(dataset)
    policy = WISEPolicy(model, stats)
    sample = dataset[0]
    observation = {key: sample[key] for key in ("images", "instruction", "state", "graph")}
    chunk = policy.predict(**observation, seed=5)
    assert chunk.shape == (config.action.horizon, config.action.dim)
    np.testing.assert_array_equal(policy.act(**observation, seed=5), chunk[0])
    np.testing.assert_array_equal(policy.act(**observation, seed=5), chunk[1])
    observation["graph"] = copy.deepcopy(observation["graph"])
    observation["graph"]["edges"] = []
    new_chunk = policy.predict(**observation, seed=5)
    np.testing.assert_array_equal(policy.act(**observation, seed=5), new_chunk[0])
    policy.reset()
    assert len(policy.queue) == 0
    result = evaluate(model, dataset, stats, max_batches=1)
    assert result["task_success_rate"] is None
    assert result["paired_graph_effect_mse"] > 0
    assert result["normalized_action_mse"] >= 0
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="violates"):
        ActionBounds([0] * 7, [0] * 7).validate(chunk)


def test_multicamera_and_partial_graph_batch(config, samples):
    samples = copy.deepcopy(samples)
    samples[0]["images"]["wrist"] = samples[0]["images"]["front"]
    samples[1]["graph"] = {"nodes": [], "edges": []}
    batch = PolicyCollator(config)(samples)
    assert batch["vlm_inputs"]["view_mask"].tolist() == [[True, True], [True, False]]
    assert not batch["graph"].node_mask[1].any()
