"""Held-out imitation error and paired graph ablation; not task-success evaluation."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_policy_model
from .data.dataset import EpisodeDataset, PolicyCollator, batch_to


@torch.inference_mode()
def evaluate(model, dataset, stats, *, batch_size=4, seed=42, max_batches=None, ablations=True):
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PolicyCollator(model.config, stats),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {
        "squared_error_normalized": 0.0,
        "absolute_error_physical": 0.0,
        "ablated_squared_error_normalized": 0.0,
        "graph_effect_squared": 0.0,
    }
    count, examples, correct, labeled = 0, 0, 0, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = batch_to(batch, device)
        targets, valid = batch["actions"], batch["action_mask"]
        noise = torch.randn(targets.shape, device=device, generator=generator)
        inputs = {key: batch[key] for key in ("vlm_inputs", "graph", "state")}
        predicted = model.sample_actions(**inputs, noise=noise, action_mask=valid)
        totals["squared_error_normalized"] += float(((predicted - targets).square() * valid).sum())
        physical = stats.denormalize_action(predicted) - stats.denormalize_action(targets)
        totals["absolute_error_physical"] += float((physical.abs() * valid).sum())
        if ablations:
            ablated = model.sample_actions(
                **inputs, noise=noise, action_mask=valid, ablate_graph=True
            )
            totals["ablated_squared_error_normalized"] += float(
                ((ablated - targets).square() * valid).sum()
            )
            totals["graph_effect_squared"] += float(((ablated - predicted).square() * valid).sum())
        context = model.encode(batch["vlm_inputs"], batch["graph"])
        has_target = batch["graph"].target.ne(-100)
        correct += int(
            (
                context.target_logits.argmax(-1)[has_target] == batch["graph"].target[has_target]
            ).sum()
        )
        labeled += int(has_target.sum())
        count += int(valid.sum())
        examples += targets.shape[0]
    if count == 0:
        raise ValueError("No valid evaluation actions")
    result = {
        "examples": examples,
        "valid_action_values": count,
        "normalized_action_mse": totals["squared_error_normalized"] / count,
        "physical_action_mae": totals["absolute_error_physical"] / count,
        "target_accuracy": correct / labeled if labeled else None,
        "target_labeled_examples": labeled,
        "task_success_rate": None,
        "note": "Offline imitation metrics; physical MAE mixes configured action units. No rollout success measured.",
    }
    if ablations:
        result.update(
            ablated_graph_normalized_mse=totals["ablated_squared_error_normalized"] / count,
            paired_graph_effect_mse=totals["graph_effect_squared"] / count,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()
    model, stats = load_policy_model(args.checkpoint, args.device, use_ema=not args.no_ema)
    data = EpisodeDataset(args.manifest, model.config.action.horizon, split=args.split)
    result = evaluate(model, data, stats, batch_size=args.batch_size, max_batches=args.max_batches)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
