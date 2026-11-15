import copy

import numpy as np
import pytest
import torch
from PIL import Image

from wise_vla.data.dataset import PolicyCollator
from wise_vla.demo import tiny_config
from wise_vla.graph import WorldGraph
from wise_vla.model import WISEVLA


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1)
    torch.manual_seed(7)


@pytest.fixture
def config():
    return tiny_config()


@pytest.fixture
def snapshot():
    world = WorldGraph()
    world.upsert_node("cube", label="red cube", node_type=1, position=(0.1, 0.2, 0.3), timestamp=0)
    world.upsert_node("tray", label="blue tray", node_type=3, position=(0.5, 0.2, 0.2), timestamp=0)
    world.set_relation("cube", "near", "tray", timestamp=0)
    return world.snapshot(0, target_id="cube")


@pytest.fixture
def samples(config, snapshot):
    sample = {
        "instruction": "Place the red cube in the tray",
        "graph": snapshot,
        "images": {"front": Image.new("RGB", (64, 64), (128, 32, 64))},
        "state": np.linspace(0.1, 0.7, 7, dtype=np.float32),
        "actions": np.arange(config.action.horizon * 7, dtype=np.float32).reshape(-1, 7) / 100,
        "action_mask": np.ones((config.action.horizon, 7), dtype=bool),
    }
    return [sample, copy.deepcopy(sample)]


@pytest.fixture
def batch(config, samples):
    return PolicyCollator(config)(samples)


@pytest.fixture
def model(config):
    return WISEVLA(config)
