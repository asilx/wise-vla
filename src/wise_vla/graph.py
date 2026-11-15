"""Typed directed multi-relational graphs, independent of any graph database."""

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .config import GraphConfig


@dataclass
class GraphBatch:
    features: Tensor  # B,N,F
    node_types: Tensor  # B,N
    node_mask: Tensor  # B,N; True means real node
    edges: Tensor  # B,N,N,R; confidence, source -> destination
    target: Tensor  # B; -100 if no target supervision
    labels: list[list[str]]
    node_ids: list[list[str]]

    def to(self, device):
        return replace(
            self,
            **{
                name: getattr(self, name).to(device)
                for name in ("features", "node_types", "node_mask", "edges", "target")
            },
        )

    def with_edges(self, edges):
        return replace(self, edges=edges)


class GraphTensorizer:
    def __init__(self, config: GraphConfig):
        self.config = config
        self.relation_ids = {name: index for index, name in enumerate(config.relations)}

    def __call__(self, snapshots: list[dict]) -> GraphBatch:
        if not snapshots:
            raise ValueError("Cannot tensorize an empty batch")
        cfg = self.config
        n = max(1, max(len(s.get("nodes", [])) for s in snapshots))
        if n > cfg.max_nodes:
            raise ValueError(f"Graph has {n} nodes; configured maximum is {cfg.max_nodes}")
        b = len(snapshots)
        features = torch.zeros(b, n, cfg.node_feature_dim)
        types = torch.zeros(b, n, dtype=torch.long)
        mask = torch.zeros(b, n, dtype=torch.bool)
        edges = torch.zeros(b, n, n, len(cfg.relations))
        targets = torch.full((b,), -100, dtype=torch.long)
        labels, node_ids = [], []
        for row, snapshot in enumerate(snapshots):
            ids = {}
            row_labels, row_ids = [""] * n, [""] * n
            for col, node in enumerate(snapshot.get("nodes", [])):
                uid = str(node["id"])
                if not uid or uid in ids:
                    raise ValueError(f"Duplicate or empty node id: {uid!r}")
                ids[uid] = col
                value = torch.tensor(node["features"], dtype=torch.float32)
                if value.shape != (cfg.node_feature_dim,) or not torch.isfinite(value).all():
                    raise ValueError(f"Node {uid}: expected {cfg.node_feature_dim} finite features")
                node_type = node.get("type", 0)
                if not isinstance(node_type, int) or not 0 <= node_type < cfg.num_node_types:
                    raise ValueError(f"Invalid node type for {uid}")
                features[row, col], types[row, col], mask[row, col] = value, node_type, True
                row_labels[col], row_ids[col] = str(node.get("label", uid)), uid
            seen = set()
            for edge in snapshot.get("edges", []):
                source, target, relation = (
                    str(edge["source"]),
                    str(edge["target"]),
                    edge["relation"],
                )
                if source not in ids or target not in ids or relation not in self.relation_ids:
                    raise ValueError(f"Unknown edge endpoint/relation: {edge}")
                if source == target:
                    raise ValueError("Self edges are not part of this graph schema")
                key = (source, target, relation)
                if key in seen:
                    raise ValueError(f"Duplicate edge: {key}")
                seen.add(key)
                confidence = float(edge.get("confidence", 1.0))
                if not 0 <= confidence <= 1:
                    raise ValueError("Edge confidence must be finite and in [0,1]")
                edges[row, ids[source], ids[target], self.relation_ids[relation]] = confidence
            if snapshot.get("target_id") is not None:
                uid = str(snapshot["target_id"])
                if uid not in ids:
                    raise ValueError("target_id must refer to an existing graph node")
                targets[row] = ids[uid]
            labels.append(row_labels)
            node_ids.append(row_ids)
        return GraphBatch(features, types, mask, edges, targets, labels, node_ids)


class WorldGraph:
    """Small timestamped world-state store for a tracker/simulator/ROS adapter.

    Coordinates are supplied by the caller in a common robot frame. This is not
    an object detector, an OWL reasoner, or a collision checker.
    """

    def __init__(self, max_age: float = 2.0):
        if max_age <= 0:
            raise ValueError("max_age must be positive")
        self.max_age = max_age
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}

    def upsert_node(
        self,
        uid,
        *,
        label,
        node_type,
        position,
        quaternion=(0, 0, 0, 1),
        size=(0, 0, 0),
        bbox=(0, 0, 0, 0),
        confidence=1.0,
        timestamp,
    ):
        if len(position) != 3 or len(quaternion) != 4 or len(size) != 3 or len(bbox) != 4:
            raise ValueError("Expected xyz(3), xyzw(4), size(3), bbox(4)")
        if not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        old = self.nodes.get(str(uid))
        if old and timestamp < old["timestamp"]:
            raise ValueError("Out-of-order node update")
        self.nodes[str(uid)] = {
            "id": str(uid),
            "label": label,
            "type": node_type,
            "features": [*position, *quaternion, *size, confidence, 0, *bbox],
            "timestamp": float(timestamp),
        }

    def set_relation(self, source, relation, target, *, confidence=1.0, timestamp):
        source, target = str(source), str(target)
        if source not in self.nodes or target not in self.nodes or source == target:
            raise ValueError("Relations require two existing, distinct nodes")
        if not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        key = (source, target, relation)
        old = self.edges.get(key)
        if old and timestamp < old["timestamp"]:
            raise ValueError("Out-of-order relation update")
        self.edges[key] = {
            "source": source,
            "target": target,
            "relation": relation,
            "confidence": confidence,
            "timestamp": float(timestamp),
        }

    def remove_node(self, uid):
        uid = str(uid)
        self.nodes.pop(uid, None)
        self.edges = {k: v for k, v in self.edges.items() if uid not in k[:2]}

    def remove_relation(self, source, relation, target):
        self.edges.pop((str(source), str(target), relation), None)

    def snapshot(self, timestamp: float, target_id: str | None = None):
        nodes = []
        for node in self.nodes.values():
            age = timestamp - node["timestamp"]
            if age < 0:
                raise ValueError("Snapshot precedes an observation")
            if age <= self.max_age:
                features = list(node["features"])
                features[11] = age / self.max_age
                nodes.append(
                    {k: v for k, v in node.items() if k not in {"timestamp", "features"}}
                    | {"features": features}
                )
        valid = {n["id"] for n in nodes}
        edges = [
            {k: v for k, v in edge.items() if k != "timestamp"}
            for edge in self.edges.values()
            if edge["source"] in valid
            and edge["target"] in valid
            and 0 <= timestamp - edge["timestamp"] <= self.max_age
        ]
        result = {"nodes": nodes, "edges": edges}
        if target_id is not None:
            if target_id not in valid:
                raise ValueError("Target is missing or stale")
            result["target_id"] = target_id
        return result
