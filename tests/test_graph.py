import copy
from dataclasses import replace

import pytest
import torch

from wise_vla.graph import GraphTensorizer, WorldGraph


def test_multi_relational_directed_edges(config, snapshot):
    snapshot["edges"].append(
        {"source": "cube", "target": "tray", "relation": "on", "confidence": 0.8}
    )
    graph = GraphTensorizer(config.graph)([snapshot])
    assert graph.edges[0, 0, 1].count_nonzero() == 2
    assert graph.edges[0, 1, 0].count_nonzero() == 0
    assert graph.target.item() == 0


@pytest.mark.parametrize(
    "mutation", ["duplicate", "unknown_relation", "nan", "bad_target", "too_many", "self_edge"]
)
def test_graph_rejects_invalid(config, snapshot, mutation):
    if mutation == "duplicate":
        snapshot["nodes"].append(copy.deepcopy(snapshot["nodes"][0]))
    elif mutation == "unknown_relation":
        snapshot["edges"][0]["relation"] = "imaginary"
    elif mutation == "nan":
        snapshot["nodes"][0]["features"][0] = float("nan")
    elif mutation == "bad_target":
        snapshot["target_id"] = "missing"
    elif mutation == "too_many":
        config.graph.max_nodes = 1
    else:
        snapshot["edges"][0]["target"] = "cube"
    with pytest.raises(ValueError):
        GraphTensorizer(config.graph)([snapshot])


def test_permutation_equivariance_and_direction(model, batch):
    model.eval()
    graph = batch["graph"]
    original = model.encode(batch["vlm_inputs"], graph)
    perm = torch.tensor([1, 0])
    shuffled = replace(
        graph,
        features=graph.features[:, perm],
        node_types=graph.node_types[:, perm],
        node_mask=graph.node_mask[:, perm],
        edges=graph.edges[:, perm][:, :, perm],
    )
    inputs = dict(batch["vlm_inputs"])
    for key in ("node_input_ids", "node_attention_mask"):
        inputs[key] = inputs[key][:, perm]
    result = model.encode(inputs, shuffled)
    torch.testing.assert_close(original.graph[:, 0], result.graph[:, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        original.graph[:, 1:][:, perm], result.graph[:, 1:], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        original.target_logits[:, perm], result.target_logits, atol=1e-5, rtol=1e-5
    )
    reversed_graph = graph.with_edges(graph.edges.transpose(1, 2))
    assert not torch.allclose(
        original.graph, model.encode(batch["vlm_inputs"], reversed_graph).graph
    )


def test_empty_and_padded_graphs_finite(config, model, samples):
    from wise_vla.data.dataset import PolicyCollator

    samples = copy.deepcopy(samples)
    samples[0]["graph"] = {"nodes": [], "edges": []}
    batch = PolicyCollator(config)(samples)
    result = model(**batch)
    assert torch.isfinite(result["loss"])
    model.eval()
    action = model.sample_actions(**{k: batch[k] for k in ("vlm_inputs", "graph", "state")})
    assert torch.isfinite(action).all()


def test_world_graph_expiry_and_order():
    world = WorldGraph(max_age=1)
    for uid in ("a", "b"):
        world.upsert_node(uid, label=uid, node_type=1, position=(0, 0, 0), timestamp=0)
    world.set_relation("a", "near", "b", timestamp=0)
    assert world.snapshot(0.5)["nodes"][0]["features"][11] == 0.5
    assert not world.snapshot(2)["nodes"]
    assert not world.snapshot(2)["edges"]
    with pytest.raises(ValueError, match="Out-of-order"):
        world.upsert_node("a", label="a", node_type=1, position=(0, 0, 0), timestamp=-1)
    with pytest.raises(ValueError, match="stale"):
        world.snapshot(2, target_id="a")
    world.remove_node("a")
    assert not world.edges
