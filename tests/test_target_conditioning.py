import copy

import pytest
import torch

from wise_vla.data.dataset import PolicyCollator
from wise_vla.model import WISEVLA


@pytest.fixture
def target_model(config):
    config.action.condition_on_target = True
    return WISEVLA(config)


def test_readout_excludes_padding_and_returns_zero_for_empty_graphs(target_model, samples):
    singleton = copy.deepcopy(samples[0])
    singleton["graph"] = {"nodes": singleton["graph"]["nodes"][:1], "edges": []}
    empty = dict(singleton, graph={"nodes": [], "edges": []})
    batch = PolicyCollator(target_model.config)([samples[0], singleton, empty])
    graph = batch["graph"]
    # Equal scores on valid nodes make the expected weighted attributes easy to check.
    with torch.no_grad():
        target_model.graph_encoder.target_query.weight.zero_()
        target_model.graph_encoder.target_query.bias.zero_()
        graph.features[~graph.node_mask] = 7
        context = target_model.encode(batch["vlm_inputs"], graph)
        expected = torch.stack([graph.features[0].mean(0), graph.features[1, 0]])
        torch.testing.assert_close(
            context.target_condition[:2], target_model.target_projection(expected)
        )
        assert context.target_condition[2].count_nonzero() == 0


def test_predicted_target_changes_actions_and_full_graph_ablation_clears_it(
    target_model, batch, monkeypatch
):
    target_model.eval()
    inputs = {key: batch[key] for key in ("vlm_inputs", "graph", "state")}
    noise = torch.randn_like(batch["actions"])

    def sample(ablated=False):
        return target_model.sample_actions(**inputs, noise=noise, steps=1, ablate_graph=ablated)

    # Intervene on the selector alone. Visual inputs, graph tokens and flow noise stay fixed.
    scores = torch.tensor([[12.0, -12.0]]).expand(len(noise), -1)
    monkeypatch.setattr(target_model.graph_encoder, "target_logits", lambda *args: scores)
    selected_first, ablated_first = sample(), sample(ablated=True)
    inputs["graph"].target.fill_(-100)
    torch.testing.assert_close(selected_first, sample(), rtol=0, atol=0)
    scores = scores.flip(-1)
    assert not torch.allclose(selected_first, sample())
    torch.testing.assert_close(ablated_first, sample(ablated=True), rtol=0, atol=0)


def test_flow_loss_trains_selector_without_target_labels(target_model, samples):
    samples[0]["graph"] = {"nodes": [], "edges": []}
    samples[1]["graph"].pop("target_id")
    batch = PolicyCollator(target_model.config)(samples)
    assert batch["graph"].target.eq(-100).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = target_model(**batch, mask_relations=False)
    losses["flow_loss"].backward()
    for layer in (
        target_model.target_projection,
        target_model.graph_encoder.target_key,
        target_model.graph_encoder.target_query,
    ):
        grad = layer.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
