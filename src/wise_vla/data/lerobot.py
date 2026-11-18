"""Import selected episodes from a local LeRobot v3.0 dataset into the WISE episode format."""

import argparse
import io
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from ..config import GraphConfig, WISEConfig
from ..graph import GraphTensorizer
from .dataset import safe_path


def episode_ids(spec: str) -> list[int]:
    """Parse comma-separated IDs and half-open ranges such as 0:5,7."""
    result = []
    for item in spec.split(","):
        if ":" in item:
            start, end = map(int, item.split(":"))
            if not 0 <= start < end:
                raise ValueError(f"Invalid episode range: {item}")
            result.extend(range(start, end))
        else:
            index = int(item)
            if index < 0:
                raise ValueError("Episode IDs must be nonnegative")
            result.append(index)
    if len(set(result)) != len(result):
        raise ValueError("An episode ID was selected more than once")
    return result


def _task_names(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    # LeRobot stores the task string as a pandas index in its official v3 snapshots.
    names = table.column_names
    name = "task" if "task" in names else "__index_level_0__"
    if name not in names or "task_index" not in names:
        raise ValueError(f"Expected task_index and task text in {path}")
    tasks = {}
    for row in table.select(["task_index", name]).to_pylist():
        index, instruction = row["task_index"], row[name]
        if index in tasks or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Invalid or duplicate task {index} in {path}")
        tasks[index] = instruction
    return tasks


def _episode_metadata(root, selected, video_keys):
    import pyarrow.dataset as ds

    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    ]
    for key in video_keys:
        columns.extend(
            f"videos/{key}/{field}"
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        )
    selection = ds.field("episode_index").isin(list(selected))
    table = ds.dataset(root / "meta/episodes", format="parquet").to_table(
        columns=columns, filter=selection
    )
    episodes = {}
    for row in table.to_pylist():
        index = row["episode_index"]
        if (
            index in episodes
            or row["length"] <= 0
            or (row["dataset_to_index"] - row["dataset_from_index"] != row["length"])
        ):
            raise ValueError(f"Invalid episode metadata for episode {index}")
        episodes[index] = row
    missing = set(selected) - episodes.keys()
    if missing:
        raise ValueError(f"Episode IDs absent from metadata: {sorted(missing)}")
    return episodes


def _graphs(directory, episode, length, tensorizer):
    if directory is None:
        return {}
    path = directory / f"episode_{episode:06d}.jsonl"
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index, graph = record["frame_index"], record["graph"]
                if type(index) is not int or not 0 <= index < length or index in result:
                    raise ValueError(f"Invalid or repeated frame_index {index}")
                if not isinstance(graph, dict) or any(
                    not isinstance(graph.get(k), list) for k in ("nodes", "edges")
                ):
                    raise ValueError("Expected a graph with nodes and edges arrays")
                tensorizer([graph])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            result[index] = graph
    return result


def _save_embedded(value, root, dest):
    if not isinstance(value, dict):
        raise ValueError("Expected a LeRobot image with bytes or a root-relative path")
    if value.get("bytes") is not None:
        source = io.BytesIO(value["bytes"])
    elif value.get("path"):
        source = safe_path(root, value["path"])
    else:
        raise ValueError("Empty LeRobot image")
    with Image.open(source) as frame:
        frame.convert("RGB").save(dest, compress_level=1)


def _write_video(path, directory, start, end, timestamps, fps):
    """Seek once to a keyframe, then match distinct frames by presentation time."""
    import av

    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError(f"Invalid episode video interval [{start}, {end}) in {path}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tick = float(stream.time_base)
        # Stay within half a frame of either clock if the camera samples faster.
        tolerance = 0.49 / max(fps, float(stream.average_rate or fps))
        boundary_eps = min(tolerance / 10, max(1e-6, tick / 2))
        if start > 0:
            container.seek(max(0, math.floor((start - tolerance) / tick)), stream=stream)
        frames = container.decode(stream)
        for index, relative in enumerate(timestamps):
            expected = start + relative
            if expected >= end:
                raise ValueError(f"Frame {index} timestamp falls outside the episode in {path}")
            for frame in frames:
                if frame.pts is None:
                    raise ValueError(f"Video frame without a presentation timestamp in {path}")
                pts = float(frame.pts * frame.time_base)
                if pts < start - boundary_eps or pts < expected - tolerance:
                    continue
                if pts >= end - boundary_eps or abs(pts - expected) > tolerance:
                    raise ValueError(
                        f"{path}: frame {index} expected {expected:.6f}s, got {pts:.6f}s"
                    )
                frame.to_image().save(directory / f"{index:06d}.png", compress_level=1)
                break
            else:
                raise ValueError(f"{path}: video ended before frame {index} at {expected:.6f}s")


def _data_rows(path, episode, columns):
    import pyarrow.dataset as ds

    # Limit live embedded-image buffers. A data shard can contain many episodes.
    scanner = ds.dataset(path, format="parquet").scanner(
        columns=columns,
        filter=ds.field("episode_index") == episode,
        batch_size=32,
        batch_readahead=1,
        fragment_readahead=1,
    )
    for batch in scanner.to_batches():
        yield from batch.to_pylist()


def _vector(value, width, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {width} finite numeric values")
    return result


def _rgb_feature(feature):
    if feature["dtype"] not in ("video", "image"):
        return False
    info = feature.get("info") or {}
    shape = feature["shape"]
    if info.get("video.is_depth_map") or len(shape) != 3:
        return False
    names = feature.get("names") or []
    if "channels" in names:
        return shape[names.index("channels")] == 3
    return shape[0] == 3 or shape[-1] == 3


def _write_episode(root, dest, info, meta, cameras, tasks, graphs):
    episode, length = meta["episode_index"], meta["length"]
    features = info["features"]
    (state_dim,) = features["observation.state"]["shape"]
    (action_dim,) = features["action"]["shape"]
    states = np.empty((length, state_dim), dtype=np.float32)
    actions = np.empty((length, action_dim), dtype=np.float32)
    timestamps = []
    image_dirs = {key: f"rgb_{i}" for i, key in enumerate(cameras)}
    for name in image_dirs.values():
        (dest / name).mkdir(parents=True)
    embedded = [key for key in cameras if features[key]["dtype"] == "image"]
    path = safe_path(
        root,
        info["data_path"].format(
            chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
        ),
    )
    columns = [
        "index",
        "frame_index",
        "timestamp",
        "task_index",
        "observation.state",
        "action",
        *embedded,
    ]
    with (dest / "observations.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(_data_rows(path, episode, columns)):
            if (
                index >= length
                or row["frame_index"] != index
                or row["index"] != meta["dataset_from_index"] + index
            ):
                raise ValueError(f"Episode {episode}: inconsistent source indices at frame {index}")
            time = row["timestamp"]
            if not math.isfinite(time) or time < 0 or (timestamps and time <= timestamps[-1]):
                raise ValueError(f"Episode {episode}: invalid timestamp at frame {index}")
            timestamps.append(time)
            states[index] = _vector(row["observation.state"], state_dim, "observation.state")
            actions[index] = _vector(row["action"], action_dim, "action")
            if row["task_index"] not in tasks:
                raise ValueError(f"Episode {episode}: unknown task_index {row['task_index']}")
            images = {key: f"{folder}/{index:06d}.png" for key, folder in image_dirs.items()}
            for key in embedded:
                _save_embedded(row[key], root, dest / images[key])
            observation = {
                "instruction": tasks[row["task_index"]],
                "timestamp": time,
                "images": images,
                "graph": graphs.get(index, {"nodes": [], "edges": []}),
            }
            stream.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
    if len(timestamps) != length:
        raise ValueError(f"Episode {episode}: expected {length} frames, found {len(timestamps)}")
    for key in cameras:
        if features[key]["dtype"] == "video":
            prefix = f"videos/{key}/"
            path = safe_path(
                root,
                info["video_path"].format(
                    video_key=key,
                    chunk_index=meta[prefix + "chunk_index"],
                    file_index=meta[prefix + "file_index"],
                ),
            )
            _write_video(
                path,
                dest / image_dirs[key],
                meta[prefix + "from_timestamp"],
                meta[prefix + "to_timestamp"],
                timestamps,
                info["fps"],
            )
    np.save(dest / "states.npy", states, allow_pickle=False)
    np.save(dest / "actions.npy", actions, allow_pickle=False)


def import_dataset(
    source: str | Path,
    output: str | Path,
    *,
    train_episodes: Iterable[int],
    val_episodes: Iterable[int] = (),
    cameras: Sequence[str] | None = None,
    graph_dir: str | Path | None = None,
    graph_config: GraphConfig | None = None,
) -> Path:
    """Publish a complete dataset by a single directory rename. Input IDs are source episode IDs."""
    root, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    if graph_dir is not None:
        graph_dir = Path(graph_dir)
        if not graph_dir.is_dir():
            raise NotADirectoryError(graph_dir)
    train, val = list(train_episodes), list(val_episodes)
    splits = {**dict.fromkeys(train, "train"), **dict.fromkeys(val, "val")}
    if (
        not train
        or len(splits) != len(train) + len(val)
        or any(type(i) is not int or i < 0 for i in splits)
    ):
        raise ValueError(
            "Use unique nonnegative episode IDs, disjoint splits and at least one training episode"
        )
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected LeRobot dataset format v3.0")
    if not math.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError("Dataset fps must be positive and finite")
    features = info["features"]
    for name in ("observation.state", "action"):
        shape = features[name]["shape"]
        if len(shape) != 1 or type(shape[0]) is not int or shape[0] < 1:
            raise ValueError(f"{name} requires a one-dimensional feature vector")
    available = [key for key, feature in features.items() if _rgb_feature(feature)]
    cameras = available if cameras is None else list(cameras)
    if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= set(available):
        raise ValueError(f"Select unique RGB camera keys from {available}")
    videos = [key for key in cameras if features[key]["dtype"] == "video"]
    episodes = _episode_metadata(root, splits, videos)
    tasks = _task_names(root / "meta/tasks.parquet")
    tensorizer = GraphTensorizer(graph_config or GraphConfig())
    metadata = {
        "format": "LeRobot v3.0",
        "robot_type": info.get("robot_type"),
        "fps": info["fps"],
        "state": features["observation.state"],
        "action": features["action"],
        "cameras": cameras,
    }
    manifest = {"schema_version": 1, "source": metadata, "episodes": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".wise-import-", dir=output.parent) as scratch:
        staged = Path(scratch) / "dataset"
        for episode, split in splits.items():
            meta = episodes[episode]
            graphs = _graphs(graph_dir, episode, meta["length"], tensorizer)
            name = f"episode_{episode:06d}"
            _write_episode(root, staged / name, info, meta, cameras, tasks, graphs)
            manifest["episodes"].append(
                {
                    "id": name,
                    "path": name,
                    "split": split,
                    "source_episode_index": episode,
                    "length": meta["length"],
                    "annotated_frames": len(graphs),
                }
            )
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        staged.rename(output)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-episodes", required=True, type=episode_ids, metavar="0:10,12")
    parser.add_argument("--val-episodes", type=episode_ids, default=[], metavar="10:12")
    parser.add_argument(
        "--cameras", nargs="+", help="Source feature keys; default: all RGB cameras"
    )
    parser.add_argument(
        "--graphs", type=Path, help="Directory of episode_NNNNNN.jsonl graph annotations"
    )
    parser.add_argument(
        "--config", type=Path, help="Use this model YAML's graph schema to validate annotations"
    )
    args = parser.parse_args()
    graph_config = WISEConfig.load(args.config).graph if args.config else None
    try:
        path = import_dataset(
            args.source,
            args.output,
            train_episodes=args.train_episodes,
            val_episodes=args.val_episodes,
            cameras=args.cameras,
            graph_dir=args.graphs,
            graph_config=graph_config,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"av", "pyarrow"}:
            raise
        parser.exit(
            2, "Install the import dependencies from the repo with pip install -e '.[lerobot]'\n"
        )
    print(path)


if __name__ == "__main__":
    main()
