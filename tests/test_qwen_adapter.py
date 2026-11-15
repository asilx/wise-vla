"""Exercise the actual Transformers Qwen vision/text implementation at tiny size.

These random weights test software compatibility, not pretrained capabilities.
The explicitly enabled integration test uses the actual cached 2B checkpoint.
"""

import os

import pytest
import torch

from wise_vla.backbones import ObservationProcessor, Qwen3VLBackbone
from wise_vla.config import VLMConfig
from wise_vla.graph import GraphTensorizer
from wise_vla.model import WISEVLA

transformers = pytest.importorskip("transformers")


def qwen_config():
    return transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": 300,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 256,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=291,
        video_token_id=292,
        vision_start_token_id=290,
        vision_end_token_id=293,
    )


def qwen_inputs(batch):
    ids = torch.tensor([[1, 290, 291, 291, 291, 291, 293, 2]]).expand(2, -1)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.randn(32, 3 * 2 * 4 * 4),
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
        "node_input_ids": batch["vlm_inputs"]["node_input_ids"],
        "node_attention_mask": batch["vlm_inputs"]["node_attention_mask"],
    }


@pytest.mark.parametrize("frozen", [True, False])
def test_real_qwen_multimodal_forward_backward(config, batch, frozen):
    backbone_model = transformers.Qwen3VLModel(qwen_config())
    config.vlm = VLMConfig(backend="qwen3_vl", dtype="float32", frozen=frozen)
    backbone = Qwen3VLBackbone(config.vlm, model=backbone_model)
    model = WISEVLA(config, backbone=backbone)
    inputs = qwen_inputs(batch)
    features = backbone(inputs)
    assert features.tokens.shape == (2, 8, 32)
    assert features.node_semantics.shape == (2, 2, 32)
    output = model(inputs, batch["graph"], batch["state"], batch["actions"])
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert model.visual_projection[0].weight.grad.abs().sum() > 0
    if frozen:
        assert all(p.grad is None for p in backbone_model.parameters())
    else:
        assert backbone_model.get_input_embeddings().weight.grad.abs().sum() > 0
        assert backbone_model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
    model.eval()
    actions = model.sample_actions(inputs, batch["graph"], batch["state"], steps=2)
    assert torch.isfinite(actions).all()


def test_bare_model_loads_conditional_generation_checkpoint(config, batch, tmp_path):
    original = transformers.Qwen3VLForConditionalGeneration(qwen_config())
    original.save_pretrained(tmp_path)
    vlm_config = VLMConfig(
        backend="qwen3_vl", model_id=str(tmp_path), dtype="float32", local_files_only=True
    )
    loaded = Qwen3VLBackbone(vlm_config)
    torch.testing.assert_close(
        original.get_input_embeddings().weight, loaded.get_input_embeddings().weight
    )
    features = loaded(qwen_inputs(batch))
    assert features.tokens.shape[-1] == 32


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("WISE_VLA_PROCESSOR_DIR"),
    reason="Set WISE_VLA_PROCESSOR_DIR to a local official processor snapshot",
)
def test_official_processor_with_reduced_qwen(config, samples):
    """Official tokenizer/template/image preprocessing, small random Qwen network."""
    location = os.environ["WISE_VLA_PROCESSOR_DIR"]
    config.vlm = VLMConfig(
        backend="qwen3_vl",
        model_id=location,
        dtype="float32",
        local_files_only=True,
        max_pixels=4096,
    )
    samples[0]["images"]["wrist"] = samples[0]["images"]["front"]
    samples[1]["instruction"] = "Move cube."
    samples[1]["graph"] = {"nodes": [], "edges": []}
    graph = GraphTensorizer(config.graph)([s["graph"] for s in samples])
    inputs = ObservationProcessor(config)(samples, graph)
    official_config = transformers.Qwen3VLConfig.from_pretrained(location, local_files_only=True)
    small = qwen_config()
    small.text_config.vocab_size = official_config.text_config.vocab_size
    small.vision_config.patch_size = official_config.vision_config.patch_size
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        setattr(small, key, getattr(official_config, key))
    backbone = Qwen3VLBackbone(config.vlm, model=transformers.Qwen3VLModel(small))
    model = WISEVLA(config, backbone=backbone).eval()
    output = model.sample_actions(inputs, graph, torch.zeros(2, 7), steps=1)
    assert output.shape == (2, 8, 7)
    assert torch.isfinite(output).all()
    assert len(inputs["image_grid_thw"]) == 3


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("WISE_VLA_PRETRAINED_TEST") != "1",
    reason="Opt in with WISE_VLA_PRETRAINED_TEST=1 and cache Qwen3-VL-2B weights",
)
def test_cached_pretrained_qwen_with_processor(config, samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.vlm = VLMConfig(
        backend="qwen3_vl",
        local_files_only=True,
        dtype="bfloat16" if device == "cuda" else "float32",
    )
    model = WISEVLA(config).to(device).eval()
    graph = GraphTensorizer(config.graph)([s["graph"] for s in samples])
    inputs = ObservationProcessor(config)(samples, graph)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.sample_actions(
        inputs, graph.to(device), torch.zeros(2, 7, device=device), steps=1
    )
    assert output.shape == (2, 8, 7)
    assert torch.isfinite(output).all()
