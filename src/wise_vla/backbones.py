"""Ready-made Qwen3-VL backbone and explicitly synthetic offline test backend."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from .config import WISEConfig
from .nn.attention import masked_mean


@dataclass
class VLMFeatures:
    tokens: Tensor
    mask: Tensor
    node_semantics: Tensor


class Backbone(nn.Module):
    frozen: bool

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            super().train(False)
        return self

    def freeze_if_requested(self):
        if self.frozen:
            self.requires_grad_(False)
            self.eval()

    def node_embeddings(self, inputs):
        ids, mask = inputs["node_input_ids"], inputs["node_attention_mask"]
        b, n, length = ids.shape
        emb = self.get_input_embeddings()(ids.reshape(b * n, length))
        return masked_mean(emb, mask.reshape(b * n, length)).reshape(b, n, -1)


class TinyBackbone(Backbone):
    """Randomly initialized byte/image encoder for CPU CI, not a pretrained VLM."""

    def __init__(self, config):
        super().__init__()
        self.output_dim, self.frozen = config.tiny_dim, config.frozen
        self.embedding = nn.Embedding(257, self.output_dim, padding_idx=0)
        self.vision = nn.Conv2d(3, self.output_dim, kernel_size=8, stride=8)
        self.freeze_if_requested()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs):
        with torch.no_grad() if self.frozen else nullcontext():
            images = inputs["pixel_values"]
            b, views, channels, height, width = images.shape
            visual = (
                self.vision(images.reshape(-1, channels, height, width)).flatten(2).transpose(1, 2)
            )
            patches = visual.shape[1]
            visual = visual.reshape(b, views * patches, -1)
            view_mask = inputs["view_mask"].repeat_interleave(patches, dim=1)
            text = self.embedding(inputs["input_ids"])
            return VLMFeatures(
                torch.cat((visual, text), 1),
                torch.cat((view_mask, inputs["attention_mask"].bool()), 1),
                self.node_embeddings(inputs),
            )


class Qwen3VLBackbone(Backbone):
    def __init__(self, config, model=None):
        super().__init__()
        self.frozen = config.frozen
        if model is None:
            from transformers import Qwen3VLForConditionalGeneration

            pretrained, loading = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_id,
                revision=config.revision,
                dtype=getattr(torch, config.dtype),
                output_loading_info=True,
                attn_implementation="sdpa",
                local_files_only=config.local_files_only,
            )
            if loading.get("missing_keys") or loading.get("mismatched_keys"):
                raise RuntimeError(f"Incomplete pretrained VLM load: {loading}")
            # Load through the checkpoint's declared class: direct bare-model loading
            # in Transformers 4.57 can silently miss the 'model.'-prefixed weights.
            model = pretrained.model
            del pretrained
        # Bare model avoids allocating vocabulary logits or every layer's hidden states.
        self.model = model
        self.output_dim = model.config.text_config.hidden_size
        self.freeze_if_requested()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, inputs):
        model_inputs = {k: v for k, v in inputs.items() if not k.startswith("node_")}
        # Parameters retain the configured VLM dtype; policy layers stay fp32/AMP.
        for key in ("pixel_values", "pixel_values_videos"):
            if key in model_inputs:
                model_inputs[key] = model_inputs[key].to(next(self.model.parameters()).dtype)
        with torch.no_grad() if self.frozen else nullcontext():
            outputs = self.model(**model_inputs, use_cache=False, return_dict=True)
            return VLMFeatures(
                outputs.last_hidden_state,
                inputs["attention_mask"].bool(),
                self.node_embeddings(inputs),
            )


def build_backbone(config):
    return TinyBackbone(config) if config.backend == "tiny" else Qwen3VLBackbone(config)


def byte_tokens(texts, limit):
    tokens = [list(text.encode("utf-8")[:limit]) or [32] for text in texts]
    length = max(map(len, tokens))
    ids = torch.zeros(len(tokens), length, dtype=torch.long)
    for i, values in enumerate(tokens):
        ids[i, : len(values)] = torch.tensor(values) + 1
    return ids, ids.ne(0)


class ObservationProcessor:
    """Accepts PIL RGB images keyed by camera name; sorts cameras deterministically."""

    def __init__(self, config: WISEConfig):
        self.config = config
        self.processor = None
        if config.vlm.backend == "qwen3_vl":
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                config.vlm.model_id,
                revision=config.vlm.revision,
                max_pixels=config.vlm.max_pixels,
                min_pixels=1024,
                local_files_only=config.vlm.local_files_only,
            )
            self.processor.tokenizer.padding_side = "left"

    def __call__(self, samples: list[dict], graph):
        cfg = self.config
        labels = [label for row in graph.labels for label in row]
        batch, nodes = graph.node_mask.shape
        for sample in samples:
            if not sample.get("instruction", "").strip() or not sample.get("images"):
                raise ValueError("Each observation needs an instruction and at least one image")
            if not all(isinstance(im, Image.Image) for im in sample["images"].values()):
                raise TypeError("images must map camera names to PIL images")
        if self.processor is None:
            ids, mask = byte_tokens([s["instruction"] for s in samples], cfg.vlm.max_text_length)
            node_ids, node_mask = byte_tokens(labels, cfg.graph.max_label_tokens)
            views = max(len(s["images"]) for s in samples)
            pixels = torch.zeros(batch, views, 3, 64, 64)
            view_mask = torch.zeros(batch, views, dtype=torch.bool)
            for b, sample in enumerate(samples):
                for v, key in enumerate(sorted(sample["images"])):
                    image = sample["images"][key].convert("RGB").resize((64, 64))
                    pixels[b, v] = (
                        torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255
                    )
                    view_mask[b, v] = True
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "pixel_values": pixels,
                "view_mask": view_mask,
            }
        else:
            prompts, images = [], []
            for sample in samples:
                content = []
                for camera in sorted(sample["images"]):
                    content.extend(
                        [{"type": "text", "text": f"Camera: {camera}."}, {"type": "image"}]
                    )
                    images.append(sample["images"][camera].convert("RGB"))
                # Do not truncate expanded multimodal tokens: that corrupts image alignment.
                instruction_ids = self.processor.tokenizer.encode(
                    sample["instruction"], add_special_tokens=False
                )
                if len(instruction_ids) > cfg.vlm.max_text_length:
                    raise ValueError("Instruction exceeds max_text_length; shorten it explicitly")
                content.append({"type": "text", "text": sample["instruction"]})
                prompts.append(
                    self.processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            inputs = dict(
                self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
            )
            encoded = self.processor.tokenizer(
                labels,
                padding=True,
                truncation=True,
                max_length=cfg.graph.max_label_tokens,
                add_special_tokens=False,
                return_tensors="pt",
            )
            node_ids, node_mask = encoded["input_ids"], encoded["attention_mask"].bool()
        inputs["node_input_ids"] = node_ids.reshape(batch, nodes, -1)
        inputs["node_attention_mask"] = node_mask.reshape(batch, nodes, -1)
        return inputs
