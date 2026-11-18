import io
import json

import numpy as np
import pytest
from PIL import Image

from wise_vla.data.dataset import EpisodeDataset, PolicyCollator
from wise_vla.data.lerobot import episode_ids, import_dataset
from wise_vla.data.normalization import fit_stats

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
av = pytest.importorskip("av")

TOP = "observation.images.top"
WRIST = "observation.images.wrist"
STILL = "observation.images.front"
DATA = "data/chunk-002/file-007.parquet"
VIDEO = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mkv"
META = "meta/episodes/chunk-000/file-000.parquet"
EMPTY = {"nodes": [], "edges": []}


def parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def video(path, length, rate=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "bgr0"
        for index in range(length):
            rgb = Image.new("RGB", (16, 16), (index * 17, 42, 80))
            for packet in stream.encode(av.VideoFrame.from_image(rgb)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    features = {
        "action": {"dtype": "float32", "shape": [1], "names": ["gripper"]},
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["pan", "tilt"]},
        TOP: {"dtype": "video", "shape": [16, 16, 3]},
        WRIST: {"dtype": "video", "shape": [16, 16, 3]},
        STILL: {"dtype": "image", "shape": [3, 16, 16], "names": ["channels", "height", "width"]},
        "depth": {"dtype": "video", "shape": [16, 16, 3], "info": {"video.is_depth_map": True}},
    }
    root.mkdir()
    (root / "meta").mkdir()
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "fps": 10,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": VIDEO,
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    parquet(
        root / "meta/tasks.parquet",
        [
            {"task_index": 4, "__index_level_0__": "Pick."},
            {"task_index": 9, "__index_level_0__": "Place."},
        ],
    )
    episodes, data = [], []
    for episode, start, length in [(0, 0, 2), (1, 2, 3)]:
        meta = {
            "episode_index": episode,
            "length": length,
            "dataset_from_index": start,
            "dataset_to_index": start + length,
            "data/chunk_index": 2,
            "data/file_index": 7,
        }
        for key, prefix_frames, file_index in [(TOP, 1, 8), (WRIST, 3, 6)]:
            camera = f"videos/{key}/"
            meta.update(
                {
                    camera + "chunk_index": 2,
                    camera + "file_index": file_index,
                    camera + "from_timestamp": (start + prefix_frames) / 10,
                    camera + "to_timestamp": (start + length + prefix_frames) / 10,
                }
            )
        episodes.append(meta)
        for frame in range(length):
            index = start + frame
            buffer = io.BytesIO()
            Image.new("RGB", (16, 16), (index, 20, 50)).save(buffer, format="PNG")
            image = {"bytes": buffer.getvalue(), "path": None}
            if index % 2:
                filename = f"still-{index}.png"
                (root / filename).write_bytes(buffer.getvalue())
                image = {"bytes": None, "path": filename}
            data.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    "index": index,
                    "task_index": 9 if index % 2 else 4,
                    "timestamp": frame / 10,
                    "observation.state": [index, -index],
                    "action": [index + 10],
                    STILL: image,
                }
            )
    parquet(root / DATA, data)
    parquet(root / META, episodes)
    for key, length, file_index in [(TOP, 6, 8), (WRIST, 8, 6)]:
        video(root / VIDEO.format(video_key=key, chunk_index=2, file_index=file_index), length)
    return root


def test_frame_alignment_and_graphs_reach_policy(source, tmp_path, config, snapshot):
    annotations = tmp_path / "graphs"
    annotations.mkdir()
    (annotations / "episode_000001.jsonl").write_text(
        json.dumps({"frame_index": 1, "graph": snapshot}) + "\n"
    )
    manifest = import_dataset(
        source,
        tmp_path / "out",
        train_episodes=[0],
        val_episodes=[1],
        graph_dir=annotations,
        graph_config=config.graph,
    )
    metadata = json.loads(manifest.read_text())
    assert metadata["source"]["state"]["names"] == ["pan", "tilt"]
    assert metadata["source"]["cameras"] == [TOP, WRIST, STILL]
    assert [e["annotated_frames"] for e in metadata["episodes"]] == [0, 1]
    train = EpisodeDataset(manifest, 3, split="train")
    val = EpisodeDataset(manifest, 3, split="val")
    assert (len(train), len(val)) == (2, 3)
    for index, sample in enumerate([train[0], train[1], val[0], val[1], val[2]]):
        assert sample["images"][TOP].getpixel((5, 0)) == ((index + 1) * 17, 42, 80)
        assert sample["images"][WRIST].getpixel((5, 0)) == ((index + 3) * 17, 42, 80)
        assert sample["images"][STILL].getpixel((5, 0)) == (index, 20, 50)
        assert sample["instruction"] == ("Place." if index % 2 else "Pick.")
        np.testing.assert_array_equal(sample["state"], [index, -index])
        assert sample["actions"][0, 0] == index + 10
        assert sample["graph"] == (snapshot if index == 3 else EMPTY)
    assert train[1]["action_mask"][:, 0].tolist() == [True, False, False]
    stats = fit_stats(train)
    assert stats.action_mean == [10.5]
    config.action.state_dim, config.action.dim = 2, 1
    batch = PolicyCollator(config, stats)([val[0], val[1]])
    assert batch["vlm_inputs"]["view_mask"].tolist() == [[True] * 3] * 2
    assert batch["graph"].node_mask.sum(1).tolist() == [0, 2]
    assert batch["graph"].target.tolist() == [-100, 0]
    assert batch["actions"].shape == (2, 3, 1)


def test_truncated_video_can_be_retried_and_output_is_protected(source, tmp_path):
    out = tmp_path / "out"
    wrist = source / VIDEO.format(video_key=WRIST, chunk_index=2, file_index=6)
    video(wrist, 7)
    with pytest.raises(ValueError, match="video ended before frame 2"):
        import_dataset(source, out, train_episodes=[0], val_episodes=[1], cameras=[WRIST])
    assert not out.exists()
    video(wrist, 8)
    manifest = import_dataset(source, out, train_episodes=[0], val_episodes=[1], cameras=[WRIST])
    assert list(EpisodeDataset(manifest, 1)[2]["images"]) == [WRIST]
    with pytest.raises(FileExistsError):
        import_dataset(source, out, train_episodes=[0])


def test_faster_video_uses_frame_at_observation_time(source, tmp_path):
    path = source / VIDEO.format(video_key=TOP, chunk_index=2, file_index=8)
    video(path, length=10, rate=40)
    manifest = import_dataset(source, tmp_path / "out", train_episodes=[0], cameras=[TOP])
    data = EpisodeDataset(manifest, 1)
    # The 10 Hz observations are at video times 0.1 and 0.2 seconds.
    assert [data[i]["images"][TOP].getpixel((0, 0))[0] for i in range(2)] == [4 * 17, 8 * 17]


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("frame_index", 0, "source indices"),
        ("timestamp", 0.15, "expected"),
        ("action", [float("nan")], "finite numeric"),
        ("task_index", 29, "unknown task_index"),
    ],
)
def test_invalid_source_frame_aborts_import(source, tmp_path, field, value, error):
    rows = pq.read_table(source / DATA).to_pylist()
    rows[1][field] = value
    parquet(source / DATA, rows)
    with pytest.raises(ValueError, match=error):
        import_dataset(source, tmp_path / "out", train_episodes=[0])
    assert not (tmp_path / "out").exists()


def test_rejects_ambiguous_splits_and_graph_rows(source, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="disjoint"):
        import_dataset(source, out, train_episodes=[0, 1], val_episodes=[1])
    with pytest.raises(ValueError, match="absent"):
        import_dataset(source, out, train_episodes=[2])
    with pytest.raises(ValueError, match="RGB"):
        import_dataset(source, out, train_episodes=[0], cameras=["depth"])
    row = json.dumps({"frame_index": 1, "graph": EMPTY}) + "\n"
    (tmp_path / "episode_000000.jsonl").write_text(row * 2)
    with pytest.raises(ValueError, match="episode_000000.jsonl:2:.*repeated"):
        import_dataset(source, out, train_episodes=[0], graph_dir=tmp_path)
    assert not out.exists()


def test_episode_id_selector():
    assert episode_ids("2,4:7") == [2, 4, 5, 6]
    for invalid in ("1:3,2", "-1", "3:3"):
        with pytest.raises(ValueError):
            episode_ids(invalid)


def test_sidecar_uses_configured_graph_schema(source, tmp_path, config, snapshot):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    snapshot["edges"][0]["relation"] = "next_to"
    (annotations / "episode_000000.jsonl").write_text(
        json.dumps({"frame_index": 0, "graph": snapshot}) + "\n"
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unknown edge endpoint/relation"):
        import_dataset(source, out, train_episodes=[0], graph_dir=annotations)
    config.graph.relations = ["next_to"]
    manifest = import_dataset(
        source,
        out,
        train_episodes=[0],
        cameras=[STILL],
        graph_dir=annotations,
        graph_config=config.graph,
    )
    sample = EpisodeDataset(manifest, 1)[0]
    assert sample["graph"]["edges"][0]["relation"] == "next_to"
