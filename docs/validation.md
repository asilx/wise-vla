# Validation record

## Predicted-target conditioning (2026-11-25)

The question was whether feeding the predicted target's numeric attributes to the action expert would help it act on the `inside` relation. The control already had the same target head, target labels during training and graph-token attention. We switched on `model.action.condition_on_target` for the intervention. This adds 1,024 weights in the tiny model. Both versions used the same 64 demonstrations, training budget (1,200 optimizer steps), EMA settings and action sampler.

Each rollout seed generates 64 test layouts. Each layout has two possible goals. Within a layout all conditions share the initial state, camera view and per-step sampling seeds. We compared two training seeds on the original test layouts and a fresh set. The training-seed-42 control used the checkpoint from the previous experiment. Its original per-episode report reproduced exactly with the updated loader.

| Training seed | Rollout seed | Success, control (full / no edges), out of 128 | Success, target-conditioned (full / no edges), out of 128 | Both goals solved with full graph (control / target-conditioned), out of 64 |
|---:|---:|---:|---:|---:|
| 42 | 2026 | 33 / 33 | 89 / 0 | 0 / 32 |
| 43 | 2026 | 32 / 32 | 91 / 0 | 0 / 34 |
| 42 | 20261125 | 40 / 40 | 91 / 0 | 0 / 33 |
| 43 | 20261125 | 40 / 40 | 92 / 3 | 0 / 34 |

With the full graph the target-conditioned policies reached the correct bin in 363 of 512 episodes (70.9%). All 149 remaining episodes reached the 16-step timeout. The controls succeeded in 145 of 512 (28.3%). Removing the relation reduced target-conditioned success to 3 of 512. On the 79 offline validation observations, all four checkpoints selected the correct target node. These measurements cover synthetic 2D reaching with the tiny backbone.

To reproduce a row, use the graph-reach recipe with its indicated `training.seed`, a separate output directory per checkpoint, and the indicated evaluation seed with `--pairs 64`. Set `model.action.condition_on_target` to `false` for the control. Layout generation uses seed 42 throughout. [Run the experiment](../README.md#measure-graph-use-in-closed-loop)

## Closed-loop graph reach (2026-11-22)

One tiny CPU policy trained for 1,200 optimizer steps on expert trajectories from 32 paired layouts. The graph provided the part's container through an `inside` relation. The target-node loss used the expert's bin ID with weight 0.25. The evaluation policy received no target ID. Generation and training used seed 42. The EMA checkpoint ran on 64 new layouts with seed 2026, with both possible goals in every layout. [Reproduction commands](../README.md#measure-graph-use-in-closed-loop)

| Check | Full graph | Same nodes with the edge removed |
|---|---:|---:|
| Correct bin reached before the wrong bin or timeout | 33 / 128 (25.8%) | 33 / 128 (25.8%) |
| Both goal assignments solved on the same layout | 0 / 64 | 0 / 64 |

The target head identified the correct bin in all 79 observations of the separate offline validation split. In the closed-loop task the action policy showed no success-rate gain from the relation in this run. An oracle regression test using the same interface reaches both goals in every tested pair when supplied the relation. These results cover synthetic 2D reaching and one trained checkpoint.

## SO-100 LeRobot import and training (2026-11-18)

The public SO-100 pick-and-place recording was imported at revision `728583b5eaf9e739a7f119e2def466fa1d552402` using PyArrow 23.0.1 and PyAV 16.1.0. Both cameras share their respective video files across episodes. [Recording and reproduction commands](data.md#reproduce-the-so-100-example)

| Check | Observed result |
|---|---|
| Episode split | Episode 0 supplied 454 training observations. Episode 1 supplied 468 validation observations. |
| Source alignment | All 922 rows matched the source states, actions, timestamps, frame indices and task instructions. |
| Camera alignment | All 1,844 exported RGB images matched a separate sequential decode from the start of the two source videos, pixel for pixel. |
| Train and evaluate | Eight CPU optimizer steps using `configs/so100_smoke.yaml`. The EMA checkpoint evaluated on all 468 validation observations, with 22,296 valid action scalars and normalized action MSE 2.527864. |
| Sparse graphs | Automated fixtures exercise exact-frame joins, absent annotations, custom relations and delivery of target labels through the policy collator. |
| Qwen preprocessing | The official Qwen3-VL processor batched the last training frame and first validation frame into four image grids and a token tensor of shape `[2, 504]`. |
| Local suite and packaging | 46 tests passed. The opt-in full 2B weight test was skipped. Ruff, wheel and source distribution builds passed. |

The SO-100 run used the randomly initialized tiny backbone and empty graphs. It measures offline imitation error. Establishing a graph benefit requires annotated demonstrations, and measuring task success requires rollouts.

## CPU model validation (2026-11-15)

Environment: Python 3.12.14, PyTorch 2.7.1+cpu, Transformers 4.57.6. CUDA was not available.

| Check | Result |
|---|---|
| Ruff on `src`, `tests`, `examples` | Passed |
| Full pytest run with official processor snapshot enabled | **36 passed, 1 skipped** |
| Full dataset → train → checkpoint → reload → evaluate → action demo | Passed, 20 optimizer steps |
| Wheel and source distribution build | Passed |
| Exact single-process CPU resume across epoch/accumulation boundaries | Passed; parameters and EMA match uninterrupted training bit for bit |
| CPU bf16 autocast forward/backward | Passed |
| Real Qwen3-VL vision/text classes at reduced random size | Forward/backward and sampling passed; both frozen and unfrozen backbones |
| Conditional-generation checkpoint → policy backbone loading | Passed; loaded token weights match the saved weights |
| Official Qwen tokenizer, chat template and image processor | Passed with multiple views, different instruction lengths and an empty graph in the batch |
| Local HTTP inference via FastAPI TestClient | Passed, including invalid-image rejection |

The HTTP test emitted a dependency deprecation warning about Starlette's httpx test-client backend. It did not affect the test result.

The remaining skipped test is `test_cached_pretrained_qwen_with_processor`: **the complete pretrained 2B weight checkpoint was not loaded or trained in this session**. Only its processor/configuration/tokenizer files were downloaded. Reduced random Qwen networks test the actual vision/text software path, not pretrained visual-language capabilities.

## Behavior covered by tests

- Finite forward/backward and flow-loss gradients into graph and action parameters.
- Frozen VLM parameters remain gradient-free; unfrozen reduced Qwen receives gradients.
- Images, language, robot state and directed relations each affect generated actions.
- Graph permutation equivariance and sensitivity to edge direction.
- Empty graphs, mixed node counts, multiple cameras and padded views.
- Masked action padding cannot influence the training loss, even if padding contains NaNs.
- Relation reconstruction removes supervised edges from encoder input.
- Target annotations are not fed into the policy encoder.
- Constant-field Euler/Heun integration with the correct noise-to-action direction.
- Deterministic seeded sampling, checkpoint round-trip and EMA restoration.
- Train/validation episode separation, chunk boundaries and per-future-timestep dimension masks.
- Statistics count each raw training timestep once.
- A fixed flow-matching batch can be overfit to below 20% of its initial loss.
- Cached actions are discarded after graph changes.

## Synthetic end-to-end run

The tiny policy has **420,437 total parameters**, including **406,037 trainable parameters**. It was trained for 20 optimizer steps on three synthetic episodes, then evaluated on one held-out synthetic episode.

| Quantity | Observed value |
|---|---:|
| Evaluation observations | 12 |
| Valid action scalar values across chunks | 476 |
| Predicted chunk shape | 8 × 7 |
| Normalized action MSE | 2.169576 |
| Graph-ablated normalized MSE | 2.164184 |
| Paired graph-effect MSE | 0.019465 |
| Target-node accuracy on synthetic labels | 1.0 |
| Robot task success | **Not measured** |

These numbers validate an executable pipeline, not a learned manipulation capability. Graph ablation was slightly better on this short synthetic run; **no graph performance improvement is established**. The synthetic target labels are trivial and do not establish real object grounding performance. The source repository excludes demo data, generated checkpoints and model weights.

## Distributed and GPU limits

An attempted two-process CPU/Gloo run could not initialize its communication transport in the local environment (`Operation not permitted` for the Gloo TCP device). The default hostname rendezvous was also unreachable. No claim of a successful local DDP training run is made.

**Hosted CI passed on Python 3.10 and 3.12**, including lint, the offline test suite, end-to-end demo, two-process CPU/Gloo training, and package builds. The verified code commit is `a133e5b6e1eafecc8bf4c6a9ae634713f4099824`; see [GitHub Actions run 43757118487](https://github.com/asilx/wise-vla/actions/runs/43757118487).

This validates the CPU distributed path on GitHub-hosted runners despite the local socket restriction. Multi-GPU/NCCL execution, full-size bf16 GPU training, throughput, memory consumption, and real-time robot execution remain unmeasured.

## Reproduce

```bash
ruff check src tests examples
pytest -q -m 'not integration'
wise-vla-demo --output runs/reproduce --train-steps 20
python -m build

# Optional official processor files, no full model weights required:
WISE_VLA_PROCESSOR_DIR=/path/to/qwen-processor pytest -q tests

# A machine permitting local process-group sockets:
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 \
  examples/ddp_smoke.py --manifest runs/reproduce/manifest.json --output runs/ddp
```
