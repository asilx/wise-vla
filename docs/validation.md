# Validation record — 2026-11-15

## Executed locally

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

The GitHub Actions workflow includes a two-process CPU smoke run on a standard hosted runner, in addition to tests and packaging. Its actual run status must be checked separately. Multi-GPU/NCCL execution, full-size bf16 GPU training, throughput, memory consumption, and real-time robot execution remain unmeasured.

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
