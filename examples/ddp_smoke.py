"""Run with torchrun --standalone --nproc-per-node=2 after generating demo data."""

import argparse

import torch.distributed as dist

from wise_vla.demo import tiny_config
from wise_vla.train import TrainingConfig, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="runs/ddp-smoke")
    parser.add_argument("--init-from")
    args = parser.parse_args()
    train = TrainingConfig(
        manifest=args.manifest,
        output_dir=args.output,
        batch_size=2,
        grad_accumulation=2,
        max_steps=4,
        warmup_steps=1,
        checkpoint_every=2,
        log_every=1,
    )
    try:
        run_training(tiny_config(), train, init_from=args.init_from)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
