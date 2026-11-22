# Training and deployment workflow

## Environment

Core installation needs PyTorch, NumPy, Pillow and PyYAML. The Qwen backend additionally uses the pinned `transformers==4.57.6` API, accelerate, safetensors and a torchvision build matching PyTorch. The code was developed with PyTorch 2.7.1. Check the validation record for executed environments.

CPU software check:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
wise-vla-demo --output runs/demo --train-steps 20
```

For GPU training, install a matching PyTorch/torchvision pair for the machine's CUDA runtime before `pip install -e '.[vlm,dev]'`. No remote compute or paid service is started by this repository. The full Qwen weights are obtained from Hugging Face on first use. To use only cached weights, set `vlm.local_files_only: true`.

## Suggested experimental sequence

1. Run the offline demo and tests on the target environment.
2. Create a small dataset from one robot/task with graph annotations that can be inspected manually.
3. Set state/action dimensions, horizon, execution steps and a consistent action schema.
4. Verify statistics and overfit a small real demonstration subset to catch alignment errors.
5. Train graph/action layers with the pretrained VLM frozen.
6. Evaluate on held-out episodes, including paired graph ablations.
7. Run closed-loop simulator evaluations before testing the robot under its independent controller and safety system.

The Qwen configuration is a starting configuration, not a measured performance or memory guarantee. Reduce batch size, image resolution, graph width/depth or action horizon when necessary. Frozen VLM weights save backward-pass memory, but its forward pass still consumes memory and compute. Cache is local to one inference chunk; persistent VLM feature caching and LoRA are not implemented.

## CLI

```bash
# Inspect or precompute statistics; training computes them automatically otherwise.
wise-vla-stats data/robot/manifest.json --split train --output normalization.json

# Train single GPU or CPU.
wise-vla-train --config configs/qwen3_vl_2b.yaml

# DistributedDataParallel with one process per GPU.
torchrun --standalone --nproc-per-node=2 -m wise_vla.train \
  --config configs/qwen3_vl_2b.yaml

# Resume optimizer, scheduler, EMA, RNG and data position.
wise-vla-train --config configs/qwen3_vl_2b.yaml \
  --resume runs/qwen3-2b/latest.pt

# Evaluate using EMA weights by default.
wise-vla-eval --checkpoint runs/qwen3-2b/latest.pt \
  --manifest data/robot/manifest.json --split val --device cuda \
  --output runs/qwen3-2b/evaluation.json
```

All relative paths in configuration files are relative to the process working directory. Commands above assume the repository root. Validation is an explicit offline command rather than a hidden training-side job. Only the latest checkpoint is retained; copy snapshots you wish to keep or use distinct output directories.

## Optimizer and precision

Training uses AdamW, configurable gradient accumulation, finite-loss checks, gradient clipping, linear warmup followed by cosine decay to 10% of the peak LR, and EMA of trainable parameters. CUDA bf16 AMP is enabled by default when `training.amp` is true; unsupported devices receive a clear error. CPU training uses fp32. There is no fp16 GradScaler path. Effective global batch size is `batch_size × grad_accumulation × world_size` except for short terminal batches.

By default only graph, projection and action layers train. `vlm.frozen: false` enables full VLM fine-tuning, which substantially changes memory/compute requirements and checkpoint size; this mode has software tests on a reduced Qwen network but has not been trained at 2B scale here. This is full fine-tuning, not LoRA. Use a suitable VLM weight dtype and optimizer setup for a large run.

## Checkpoints and reproducibility

`latest.pt` is atomically replaced and contains:

- Model architecture and resolved pretrained-backbone revision.
- Trainable policy weights, or the full model if the backbone is trainable.
- The complete random tiny backbone for offline runs.
- Training-split normalization statistics.
- Optimizer, scheduler and EMA states.
- Per-rank PyTorch RNG states, sampler epoch, batch offset and optimizer step.
- World size, training settings, manifest checksum and PyTorch version.

Frozen pretrained weights are not duplicated; they must be available at the saved Hub revision or unchanged local model directory. Loading uses `torch.load(..., weights_only=True)` and validates expected parameter keys. Resume resolves the saved revision before loading the backbone.

Resume requires the same training plan and world size. You may change the output directory and logging/checkpoint cadence, but not batch size, accumulation, seed or the scheduler's total step budget. A manifest checksum detects manifest edits; it does not hash all image/array contents. Keep the entire dataset immutable. The exact CPU resume test uses `num_workers=0`, unchanged software and deterministic preprocessing. Different CUDA kernels, hardware or worker-side stochastic transformations can change results; bitwise GPU reproducibility is not claimed.

DistributedSampler may repeat a few examples to balance ranks when the dataset size is not divisible by world size. `max_steps` counts optimizer steps. A checkpoint is written only after a completed accumulation/update cycle. Gradients from an interrupted partial cycle are intentionally not retained.

## Evaluation interpretation

Reported metrics include normalized action MSE, denormalized action MAE, optional target-node accuracy, graph-ablated MSE and a paired graph-effect MSE. Baseline and ablated sampling use the same Gaussian noise. The ablation removes the structured graph context supplied to action blocks; it retains VLM inputs and the rest of the trained model. It tests dependency and sensitivity, not a retrained graph-free baseline.

Physical MAE averages the configured action channels and can mix units. Interpret it together with per-channel robot conventions. Overlapping chunks mean later timesteps can be represented in several evaluation samples. The metric is chunk imitation error, not independent episode success. A nonzero graph-effect metric is not evidence of a useful graph effect. `task_success_rate` is deliberately null until an external rollout evaluator supplies actual task outcomes.

## Robot adapter responsibilities

Use `WISEPolicy.predict()` for a full denormalized chunk or `WISEPolicy.act()` for queued receding-horizon actions. Inputs are PIL images, instruction text, a state vector and a graph snapshot. The wrapper is stateful and must be reset between episodes. It replans on graph/instruction changes, but the controller owns timing and can force replanning after other events.

An optional `ActionBounds` object rejects chunks outside configured channel limits. It does not model joint dynamics, velocity/acceleration limits, collision geometry, reachability or functional safety. Keep those checks in the robot controller. The local HTTP example is a single-policy inference service with serialized model access; it is not an authenticated public service or a robot driver.

## Tests

```bash
pip install -e '.[dev]' transformers==4.57.6
ruff check src tests examples
pytest -q -m 'not integration'
```

Core tests run without downloading a model. The Qwen tests instantiate the actual Transformers vision/text classes with small random weights and verify multimodal forward/backward and checkpoint prefix loading. They are skipped if Transformers is absent.

Two explicitly enabled integration tests are available:

```bash
# A directory holding the official tokenizer, chat template and processor files.
WISE_VLA_PROCESSOR_DIR=/path/to/official-processor pytest -q tests/test_qwen_adapter.py

# Actual cached Qwen3-VL-2B weights; potentially substantial RAM/VRAM.
WISE_VLA_PRETRAINED_TEST=1 pytest -q tests/test_qwen_adapter.py -k cached_pretrained
```

The processor test uses official preprocessing with a reduced randomly initialized Qwen network. The cached-pretrained test is the only test that loads the actual 2B pretrained weights. Consult `docs/validation.md` before assuming either was executed on a particular machine.

## Graph reach protocol

The task state is the cursor's absolute `(x, y)` position in a unit square. Actions are displacements in the same coordinates. The simulator limits the Euclidean step length to 0.08, clips positions to the workspace, then renders the updated view. The first bin center within radius 0.065 ends the episode. The correct bin gives success, the other gives a wrong-bin outcome, and 16 steps without either gives a timeout.

The expert moves directly toward the bin containing the hidden part. Paired demonstrations share the starting position, bin geometry, image and instruction. The saved `target_id` supplies the trainer's auxiliary classification loss. The online policy receives only the three nodes and, in the full condition, the directed `inside` edge. Relation masking is disabled for this task because that single edge specifies the goal.

The evaluator resets the action queue for every rollout and uses the same per-step sampling seeds for all four trials in a layout. Train, validation and test layouts use separate random streams. With both possible goals, an edge-free policy following the same trajectory can succeed on at most one of the two episodes. A score of 1 for `both_goals_reached_rate` requires reaching each destination when its turn comes. The JSON report records each trial's condition, goal, outcome and step count. [Run the CPU example](../README.md#measure-graph-use-in-closed-loop)
