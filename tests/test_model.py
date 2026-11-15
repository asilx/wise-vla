import copy

import pytest
import torch
from torch import nn

from wise_vla.config import WISEConfig


def test_flow_gradients_reach_graph_and_expert_not_frozen_vlm(model, batch):
    model.train()
    assert not model.backbone.training
    output = model(**batch, mask_relations=False)
    output["flow_loss"].backward()
    for parameter in (
        model.action_expert.blocks[0].graph_attn.k.weight,
        model.graph_encoder.blocks[0].forward_bias,
        model.graph_encoder.features[0].weight,
        model.semantic_projection[0].weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_auxiliary_objectives_backpropagate(model, batch):
    model.config.graph.relation_mask_probability = 1.0
    output = model(**batch)
    assert output["relation_loss"] > 0 and output["target_loss"] > 0
    (output["relation_loss"] + output["target_loss"]).backward()
    assert model.graph_encoder.relation_head[-1].weight.grad.abs().sum() > 0
    assert model.graph_encoder.target_query.weight.grad.abs().sum() > 0


def test_masked_padding_does_not_change_loss(model, batch):
    model.eval()
    batch["action_mask"][:, -3:] = False
    noise = torch.randn_like(batch["actions"])
    time = torch.tensor([0.2, 0.8])
    first = model(**batch, noise=noise, time=time)["flow_loss"]
    batch["actions"][:, -3:] = float("nan")
    noise[:, -3:] = float("nan")
    second = model(**batch, noise=noise, time=time)["flow_loss"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_each_condition_affects_actions(model, batch):
    model.eval()
    inputs = {key: batch[key] for key in ("vlm_inputs", "graph", "state")}
    noise = torch.randn_like(batch["actions"])
    baseline = model.sample_actions(**inputs, noise=noise, steps=2)
    ablated = model.sample_actions(**inputs, noise=noise, steps=2, ablate_graph=True)
    assert not torch.allclose(baseline, ablated)
    for condition in ("image", "language", "state", "relations"):
        changed = copy.deepcopy(inputs)
        if condition == "image":
            changed["vlm_inputs"]["pixel_values"] *= 0
        elif condition == "language":
            changed["vlm_inputs"]["input_ids"] = changed["vlm_inputs"]["input_ids"].flip(1) + 1
            # Tiny byte-mean context is order invariant: alter token identities too.
        elif condition == "state":
            changed["state"] += 1
        else:
            changed["graph"].edges *= 0
        result = model.sample_actions(**changed, noise=noise, steps=2)
        assert not torch.allclose(baseline, result), condition


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_solver_orientation_constant_field(model, batch, solver):
    class ConstantVelocity(nn.Module):
        def forward(self, x, time, state, context, valid):
            return torch.full_like(x, 2).masked_fill(~valid, 0)

    model.action_expert = ConstantVelocity()
    model.eval()
    mask = batch["action_mask"]
    mask[:, -2:] = False
    noise = torch.ones_like(batch["actions"])
    result = model.sample_actions(
        **{k: batch[k] for k in ("vlm_inputs", "graph", "state")},
        noise=noise,
        action_mask=mask,
        steps=5,
        solver=solver,
    )
    torch.testing.assert_close(result[mask], torch.full_like(result[mask], 3))
    assert result[~mask].count_nonzero() == 0


def test_seeded_sampling_and_validation(model, batch):
    inputs = {k: batch[k] for k in ("vlm_inputs", "graph", "state")}
    with pytest.raises(RuntimeError, match="eval"):
        model.sample_actions(**inputs)
    model.eval()
    a = model.sample_actions(**inputs, generator=torch.Generator().manual_seed(5))
    b = model.sample_actions(**inputs, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model.sample_actions(**inputs, steps=0)
    with pytest.raises(ValueError, match="valid action"):
        model(**{**batch, "action_mask": torch.zeros_like(batch["action_mask"])})


def test_config_strict_and_roundtrip(config, tmp_path):
    path = tmp_path / "config.yaml"
    config.save(path)
    assert WISEConfig.load(path) == config
    with pytest.raises(TypeError):
        WISEConfig(unknown_setting=True)
    with pytest.raises(ValueError):
        WISEConfig(hidden_dim=33)


def test_bfloat16_autocast_forward_backward(model, batch):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(**batch)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_target_annotation_is_not_a_model_input(model, batch):
    model.eval()
    before = model.encode(batch["vlm_inputs"], batch["graph"])
    batch["graph"].target[:] = 1
    after = model.encode(batch["vlm_inputs"], batch["graph"])
    torch.testing.assert_close(before.graph, after.graph, rtol=0, atol=0)
    torch.testing.assert_close(before.target_logits, after.target_logits, rtol=0, atol=0)


def test_relation_labels_hidden_from_encoder(model, batch):
    model.config.graph.relation_mask_probability = 1
    observed = []
    handle = model.graph_encoder.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].edges.clone())
    )
    original = batch["graph"].edges.clone()
    try:
        result = model(**batch)
    finally:
        handle.remove()
    assert observed[0].count_nonzero() == 0
    assert result["relation_loss"] > 0
    torch.testing.assert_close(batch["graph"].edges, original)
