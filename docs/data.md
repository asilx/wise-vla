# Dataset and graph contract

For collection design, supervision sources, episode boundaries at expert takeover and split design across training stages, see [what data WISE-VLA needs](pretraining-and-posttraining-data.md).

## On-disk layout

A dataset root contains `manifest.json` and one directory per episode. Each episode has `states.npy`, `actions.npy`, `observations.jsonl`, and referenced RGB image files. NumPy arrays are opened with `allow_pickle=False` and memory-mapped; observation metadata is loaded into memory. This simple format is intended as an explicit interchange layer, not a petabyte-scale streaming format.

```json
{
  "schema_version": 1,
  "episodes": [
    {"id": "episode_000", "path": "episode_000", "split": "train"},
    {"id": "episode_001", "path": "episode_001", "split": "val"}
  ]
}
```

- `states.npy`: numeric array `[T, state_dim]`.
- `actions.npy`: numeric array `[T, action_dim]`.
- `observations.jsonl`: exactly `T` JSON objects, in timestep order.
- All observations at index `t` precede action `actions[t]`; do not insert future state or future graph information into an observation.
- Image and optional graph-file paths are relative to the episode directory and cannot escape it.
- Episodes must have unique IDs. Train/validation separation is by episode, not by randomly splitting overlapping chunks.

One observation line, shown pretty-printed for readability:

```json
{
  "instruction": "Place the red cube in the blue tray.",
  "images": {"front": "front_0000.png", "wrist": "wrist_0000.png"},
  "graph": {
    "nodes": [
      {"id": "cube_1", "label": "red cube", "type": 1,
       "features": [0.4, 0.1, 0.8, 0, 0, 0, 1, 0.04, 0.04, 0.04, 0.98, 0, 0.2, 0.3, 0.4, 0.6]},
      {"id": "tray_1", "label": "blue tray", "type": 3,
       "features": [0.6, 0.1, 0.8, 0, 0, 0, 1, 0.2, 0.2, 0.03, 0.97, 0, 0.6, 0.3, 0.9, 0.7]}
    ],
    "edges": [
      {"source": "cube_1", "target": "tray_1", "relation": "near", "confidence": 0.9}
    ],
    "target_id": "cube_1"
  },
  "action_dim_mask": [true, true, true, true, true, true, true]
}
```

`graph` may instead be a relative path to a JSON snapshot. Missing `graph` means an empty graph. `target_id` is optional **training supervision**, not a required inference input. `action_dim_mask` is optional; it specifies valid action dimensions at this timestep. The loader gathers these masks for each future step in a chunk and masks the episode tail. Every sample must retain at least one valid action.

## Numeric node features

The default graph feature width is 16. `WorldGraph.upsert_node()` produces this ordering:

| Indices | Value | Convention |
|---|---|---|
| 0–2 | Position | x, y, z in meters, common robot frame |
| 3–6 | Orientation | Quaternion x, y, z, w; normalize before ingestion |
| 7–9 | Extents | x, y, z in meters |
| 10 | Confidence | [0,1] |
| 11 | Observation age | elapsed seconds / configured maximum age |
| 12–15 | Bounding box | x1, y1, x2, y2 normalized to [0,1] in the agreed reference camera |

An all-zero box means unavailable; it is a numeric cue, not a hard ROI mask. Keep the reference camera fixed across training and inference. Use quaternion `(0,0,0,1)` only when identity orientation is a valid observation/default for your task. The tensorizer validates dimensions and finite numbers; it cannot certify frame alignment or physical correctness.

Default node-type convention: `0=unknown`, `1=object`, `2=robot`, `3=location/support`, `4=tool`, `5=person`, `6=task`, `7=other`. The embedding table is configurable. The model does not infer these types automatically.

Default predicates: `on`, `inside`, `near`, `holding`, `reachable`, `part_of`, `target`, `blocked_by`. They are exact case-sensitive strings. Multiple distinct predicates may connect a node pair. Duplicate predicates on one pair, unknown endpoints, self edges and invalid confidences are rejected. Predicate orientation must stay consistent: e.g. `tool on bench`, `robot holding tool`, `object inside container`.

The default relation-reconstruction loss assumes unannotated predicates are absent. If graphs are incomplete rather than exhaustive, disable that loss with `model.graph.relation_loss_weight: 0`. Graph conditioning remains active. Do not turn label-only target annotations into input `target` edges during evaluation.

## State/action semantics

The default width 7 is a dimension setting, not a guaranteed robot convention. A single-arm adapter might define actions as Cartesian delta position (3), rotation-vector delta (3), and gripper aperture (1), but the trainer makes no such assumption. Joint position, joint velocity, absolute pose or other control schemas require matching demonstration data and a controller with identical semantics.

Record the following alongside each dataset: ordered channel names, units, coordinate frame, delta/absolute convention, gripper representation, control frequency, camera order/calibration, robot model and tool configuration. All episodes used for one checkpoint must share this schema. Do not mix different control modes merely because their vector sizes match.

Statistics are fitted on the training split only and count each raw timestep once. Padded action dimensions are excluded. Every dimension needs at least one valid training observation. Standard deviations have a `1e-3` floor; this is affine z-score normalization, not quantile clipping. Graph geometry keeps its documented physical units.

## LeRobot v3.0 import

`wise-vla-import-lerobot` reads a local LeRobot v3.0 dataset with per-episode metadata, task text, one-dimensional `observation.state` and `action` features, and at least one RGB camera. It accepts shared video files and Parquet image features containing bytes or a path relative to the source root. The resulting manifest works with the existing WISE training, evaluation, and inference commands. [LeRobot v3 format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

### Reproduce the SO-100 example

From the repository root, use the Hugging Face CLI to download the shards containing episodes 0 and 1 of the public pick-and-place recording. Both episodes share the data and video files. The pinned revision contains the camera views `observation.images.top` and `observation.images.wrist`. [Source recording](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/tree/728583b5eaf9e739a7f119e2def466fa1d552402)

```bash
pip install -e '.[lerobot]' huggingface_hub
hf download lerobot/svla_so100_pickplace meta/info.json meta/tasks.parquet \
  meta/episodes/chunk-000/file-000.parquet data/chunk-000/file-000.parquet \
  videos/observation.images.top/chunk-000/file-000.mp4 \
  videos/observation.images.wrist/chunk-000/file-000.mp4 \
  --repo-type dataset --revision 728583b5eaf9e739a7f119e2def466fa1d552402 \
  --local-dir data/so100-source
wise-vla-import-lerobot data/so100-source data/so100-wise --train-episodes 0 --val-episodes 1
wise-vla-train --config configs/so100_smoke.yaml
wise-vla-eval --checkpoint runs/so100-smoke/latest.pt \
  --manifest data/so100-wise/manifest.json --split val --device cpu --batch-size 8
```

The example runs eight optimizer steps with the offline backbone and a horizon of eight six-channel actions. For a different robot, use its state and action widths in the model configuration. The importer writes the source feature descriptions, camera keys and frame rate to `manifest.json`, and saves raw state and action values as float32 in source channel order. Control semantics and physical units come from the recording.

### Select episodes and cameras

- `--train-episodes 0:8,12 --val-episodes 8:10` assigns source IDs 0–7 and 12 to training, 8 and 9 to validation. Ranges exclude the upper endpoint. IDs must be unique across both lists.
- By default, all declared RGB cameras are imported. For a single view, pass e.g. `--cameras observation.images.wrist`. Each observation keeps the original camera keys. The policy orders views by key when batching.
- The importer reads each camera's own shard and episode start time, adds the row's episode-relative timestamp and checks the decoded video presentation time. It writes RGB PNGs and preserves the original timestamp in `observations.jsonl`. Missing or misaligned frames abort the import.
- Use a new output directory. The final manifest and episode directories appear together once every selected episode passes validation. Decoded PNGs can occupy more disk space than the source videos.

### Attach knowledge-graph snapshots

Pass `--graphs annotations` during import. Create `annotations/episode_000001.jsonl` for source episode 1, with one record per annotated frame. This example marks the cube as the supervised target at frame 12. The 16 features follow the ordering above.

```json
{"frame_index":12,"graph":{"nodes":[{"id":"cube","label":"cube","type":1,"features":[0.1,0.2,0.3,0,0,0,1,0.04,0.04,0.04,0.98,0,0.2,0.3,0.4,0.6]}],"edges":[],"target_id":"cube"}}
```

Frame indices are zero-based within the source episode. Supply geometry and relations observed at that timestep. A frame without an annotation receives an empty graph, including frames in episodes with no annotation file. Each annotation applies to exactly one frame. The importer records the number of annotated frames per episode and validates every supplied snapshot with the policy's graph tensorizer. Pass `--config path/to/model.yaml` to use a custom graph schema. See [a runtime graph construction example](../examples/world_graph.py).

## Temporal limitations

The policy consumes a current observation and a current graph snapshot. `WorldGraph` stores recent entity state and expiry, but the neural encoder is not a recurrent temporal graph network. Historical video, recurrent hidden memory, and offline observation augmentation are not implemented. `WorldGraph` timestamps must be monotonic and use the same clock as its snapshots; expire and prune long-lived stores as appropriate for the tracker.
