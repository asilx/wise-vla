"""Populate a live graph from externally grounded object observations."""

import json

from wise_vla.graph import WorldGraph

world = WorldGraph(max_age=1.0)
world.upsert_node(
    "tool_1",
    label="torque wrench",
    node_type=4,
    position=(0.4, 0.1, 0.8),
    size=(0.25, 0.04, 0.03),
    bbox=(0.2, 0.3, 0.5, 0.6),
    confidence=0.97,
    timestamp=12.0,
)
world.upsert_node(
    "bench_1",
    label="assembly bench",
    node_type=3,
    position=(0.4, 0.0, 0.75),
    size=(1.2, 0.8, 0.75),
    timestamp=12.0,
)
world.set_relation("tool_1", "on", "bench_1", confidence=0.95, timestamp=12.0)
print(json.dumps(world.snapshot(12.1), indent=2))
