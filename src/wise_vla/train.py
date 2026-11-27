"""Single-process or torchrun DDP training with bf16 AMP, EMA and resumable sampling."""

import argparse
import hashlib
import json
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import (
    EMA,
    check_model_config,
    checkpoint_sha256,
    read_checkpoint,
    restore_weights,
    save_checkpoint,
)
from .config import WISEConfig
from .data.dataset import EpisodeDataset, PolicyCollator, batch_to
from .data.normalization import NormalizationStats, fit_stats
from .model import WISEVLA


@dataclass
class TrainingConfig:
    manifest: str = "runs/demo/manifest.json"
    output_dir: str = "runs/wise-vla"
    train_split: str = "train"
    batch_size: int = 8
    max_steps: int = 10000
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_accumulation: int = 1
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    checkpoint_every: int = 1000
    log_every: int = 10
    num_workers: int = 0
    seed: int = 42
    amp: bool = True
    normalization: str | None = None

    def __post_init__(self):
        if (
            min(
                self.batch_size,
                self.max_steps,
                self.grad_accumulation,
                self.checkpoint_every,
                self.log_every,
            )
            < 1
        ):
            raise ValueError("Training counts must be positive")
        if self.num_workers < 0 or self.warmup_steps < 0:
            raise ValueError("Workers/warmup must be nonnegative")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid EMA decay")


def rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _save_run_settings(output, config, train, stats, *, new_stage, rank, distributed):
    error = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=True)
            if new_stage and any(
                (output / name).exists() for name in ("latest.pt", "metrics.jsonl")
            ):
                raise FileExistsError(output)
            # An exclusive create also protects two new stages racing for the same directory.
            with (output / "training_config.json").open("x" if new_stage else "w") as stream:
                json.dump(asdict(train), stream, indent=2)
            config.save(output / "model_config.yaml")
            stats.save(output / "normalization.json")
        except OSError as exc:
            if new_stage and isinstance(exc, FileExistsError):
                error[0] = "Use a separate, unused output directory for the new stage"
            else:
                error[0] = f"Could not write training settings in {output}: {exc}"
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise ValueError(error[0])


def run_training(
    config: WISEConfig, train: TrainingConfig, *, resume=None, init_from=None, stop_after=None
):
    if resume is not None and init_from is not None:
        raise ValueError("Choose either resume or init_from")
    output = Path(train.output_dir)
    latest = output / "latest.pt"
    new_stage = init_from is not None
    origin = None
    if new_stage:
        if Path(init_from).resolve().parent == output.resolve():
            raise ValueError("Use a separate, unused output directory for the new stage")
        with Path(init_from).open("rb") as stream:
            digest = checkpoint_sha256(stream)
            payload = read_checkpoint(stream)
        origin = {
            "checkpoint_sha256": digest,
            "step": payload["trainer"].get("step"),
            "weights": "ema" if payload["ema"] is not None else "model",
        }
    else:
        payload = read_checkpoint(resume) if resume is not None else None
        if payload:
            origin = payload["trainer"].get("init_from")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed and new_stage:
        sources = [None] * world
        dist.all_gather_object(sources, origin["checkpoint_sha256"])
        if len(set(sources)) != 1:
            raise ValueError("All ranks must initialize from the same source checkpoint")
    if payload:
        if config.vlm.revision == "main":
            # Restore the resolved model version before performing any Hub download.
            config.vlm.revision = payload["config"]["vlm"]["revision"]
        check_model_config(config, payload, new_stage=new_stage)
    torch.manual_seed(train.seed)  # identical initial parameters across ranks
    model = WISEVLA(config).to(device)
    if payload:
        restore_weights(model, payload, use_ema=new_stage, new_stage=new_stage)
    dataset = EpisodeDataset(train.manifest, config.action.horizon, split=train.train_split)
    if len(dataset) < world:
        raise ValueError("Dataset must have at least world_size timesteps")
    for episode in dataset.episodes:
        if (episode["state_dim"], episode["action_dim"]) != (
            config.action.state_dim,
            config.action.dim,
        ):
            raise ValueError(
                f"State/action dimensions do not match the policy in episode {episode['id']}"
            )
    stats = (
        NormalizationStats(**payload["normalization"])
        if payload
        else NormalizationStats.load(train.normalization)
        if train.normalization
        else fit_stats(dataset)
    )
    if new_stage and train.normalization and NormalizationStats.load(train.normalization) != stats:
        raise ValueError("The new stage must use the source checkpoint's normalization")
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=train.seed, drop_last=False
    )
    loader = DataLoader(
        dataset,
        batch_size=train.batch_size,
        sampler=sampler,
        num_workers=train.num_workers,
        collate_fn=PolicyCollator(config, stats),
        generator=torch.Generator().manual_seed(train.seed),
        pin_memory=False,
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=train.learning_rate, weight_decay=train.weight_decay
    )

    def schedule(step):
        if step < train.warmup_steps:
            return (step + 1) / max(1, train.warmup_steps)
        progress = (step - train.warmup_steps) / max(1, train.max_steps - train.warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * min(progress, 1))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = EMA(model, train.ema_decay)
    epoch, offset, step = 0, 0, 0
    manifest_hash = hashlib.sha256(Path(train.manifest).read_bytes()).hexdigest()
    signature = {
        k: v
        for k, v in asdict(train).items()
        if k not in {"output_dir", "checkpoint_every", "log_every", "normalization"}
    }
    if resume is not None:
        previous = payload["trainer"]
        if previous["world_size"] != world or previous["training_signature"] != signature:
            raise ValueError("Resume requires matching world size and training settings")
        if previous["manifest_sha256"] != manifest_hash:
            raise ValueError("Dataset manifest changed since checkpoint")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema.shadow = {n: p.to(device) for n, p in payload["ema"].items()}
        epoch, offset, step = previous["epoch"], previous["batch_offset"], previous["step"]
    _save_run_settings(
        output, config, train, stats, new_stage=new_stage, rank=rank, distributed=distributed
    )
    if rank == 0:
        print(
            json.dumps(
                {"parameters": model.parameter_counts(), "device": str(device), "world_size": world}
            ),
            flush=True,
        )
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if distributed
        else model
    )
    torch.manual_seed(train.seed + rank)
    if resume is not None and payload["rng_states"]:
        restore_rng(payload["rng_states"][rank], device)
    del payload
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    amp = train.amp and device.type == "cuda"
    if amp and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 AMP requires a supported GPU; set training.amp=false")
    limit = train.max_steps if stop_after is None else min(stop_after, train.max_steps)

    def checkpoint():
        local_rng = rng_state(device)
        states = [None] * world if distributed else [local_rng]
        if distributed:
            dist.all_gather_object(states, local_rng)
        if rank == 0:
            save_checkpoint(
                latest,
                model,
                stats=stats,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                trainer_state={
                    "step": step,
                    "epoch": epoch,
                    "batch_offset": offset,
                    "world_size": world,
                    "training_signature": signature,
                    "manifest_sha256": manifest_hash,
                    **({"init_from": origin} if origin is not None else {}),
                },
                rng_states=states,
            )
        if distributed:
            dist.barrier()

    while step < limit:
        metrics = {}
        for micro in range(train.grad_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch, offset = epoch + 1, 0
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            offset += 1
            batch = batch_to(batch, device)
            sync_context = (
                wrapped.no_sync()
                if distributed and micro < train.grad_accumulation - 1
                else nullcontext()
            )
            with (
                sync_context,
                torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp),
            ):
                result = wrapped(**batch)
                loss = result["loss"] / train.grad_accumulation
            finite = torch.isfinite(loss).to(torch.int32)
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite:
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            for key in ("loss", "flow_loss", "relation_loss", "target_loss"):
                metrics[key] = metrics.get(key, 0) + result[key].detach() / train.grad_accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, train.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        step += 1
        if step % train.log_every == 0 or step == limit:
            if distributed:
                for value in metrics.values():
                    dist.all_reduce(value)
                    value /= world
            if rank == 0:
                row = {key: float(value) for key, value in metrics.items()}
                row.update(
                    step=step,
                    epoch=epoch,
                    lr=scheduler.get_last_lr()[0],
                    grad_norm=float(grad_norm),
                )
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        if step % train.checkpoint_every == 0 or step == limit:
            checkpoint()
    if step >= limit and not latest.exists() and rank == 0:
        raise ValueError("No optimizer steps requested and no output checkpoint exists")
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--resume", help="Continue this training run from its checkpoint")
    inputs.add_argument("--init-from", help="Start a fresh stage from a WISE policy checkpoint")
    args = parser.parse_args()
    values = yaml.safe_load(Path(args.config).read_text())
    try:
        run_training(
            WISEConfig(**values["model"]),
            TrainingConfig(**values.get("training", {})),
            resume=args.resume,
            init_from=args.init_from,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
