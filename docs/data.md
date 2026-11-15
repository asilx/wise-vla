# Dataset and graph contract

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

## Importing existing robotics data

The format is deliberately independent of a specific simulator or LeRobot version. An importer should:

1. Split complete episodes into train/validation sets.
2. Export aligned numeric arrays and RGB frames.
3. Export one observation record per timestep.
4. Populate graphs from simulator ground truth, perception/tracking, human annotation, or an external knowledge system.
5. Verify frame, unit and time alignment before training.

No automatic LeRobot, RLDS, ROS or Isaac importer is included in v0.1. The source arrays can be written with `np.save(..., allow_pickle=False)`. `wise-vla-demo` creates a complete runnable example of this contract. `examples/world_graph.py` illustrates the runtime graph API. Supplying an empty graph is supported for a VLM-only starting dataset, but it does not teach graph use.

## Temporal limitations

The policy consumes a current observation and a current graph snapshot. `WorldGraph` stores recent entity state and expiry, but the neural encoder is not a recurrent temporal graph network. Historical video, recurrent hidden memory, and offline observation augmentation are not implemented. `WorldGraph` timestamps must be monotonic and use the same clock as its snapshots; expire and prune long-lived stores as appropriate for the tracker.
