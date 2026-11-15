import base64
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_local_http_prediction_contract(config, model, samples):
    from fastapi.testclient import TestClient

    from wise_vla.data.normalization import NormalizationStats
    from wise_vla.policy import WISEPolicy

    path = Path(__file__).parents[1] / "examples" / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("serve_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = NormalizationStats([0.0] * 7, [1.0] * 7, [0.0] * 7, [1.0] * 7)
    client = TestClient(module.create_app(WISEPolicy(model, stats)))
    assert client.get("/health").json()["synthetic_backbone"]
    encoded = io.BytesIO()
    samples[0]["images"]["front"].save(encoded, format="PNG")
    payload = {
        "instruction": samples[0]["instruction"],
        "state": [0.0] * 7,
        "graph": samples[0]["graph"],
        "seed": 7,
        "images": {"front": base64.b64encode(encoded.getvalue()).decode()},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    actions = np.array(response.json()["actions"])
    assert actions.shape == (config.action.horizon, config.action.dim)
    assert np.isfinite(actions).all()
    payload["images"]["front"] = "not-base64"
    assert client.post("/predict", json=payload).status_code == 422
