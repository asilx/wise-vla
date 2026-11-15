"""Optional local HTTP inference. Start with a trained checkpoint; no robot driver included."""

import argparse
import base64
import binascii
import io
import threading

from PIL import Image, UnidentifiedImageError

from wise_vla.policy import WISEPolicy


def create_app(policy):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class Observation(BaseModel):
        instruction: str = Field(min_length=1, max_length=4096)
        images: dict[str, str]  # camera name -> base64 PNG/JPEG (no data-URL prefix)
        state: list[float]
        graph: dict
        seed: int | None = None

    app = FastAPI(title="WISE-VLA inference", version="0.1.0")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "ready": True,
            "backbone": policy.model.config.vlm.backend,
            "synthetic_backbone": policy.model.config.vlm.backend == "tiny",
        }

    @app.post("/predict")
    def predict(observation: Observation):
        if not 1 <= len(observation.images) <= 8:
            raise HTTPException(status_code=422, detail="Expected 1–8 camera images")
        images = {}
        try:
            for name, encoded in observation.images.items():
                if len(encoded) > 8_000_000:
                    raise ValueError("Image exceeds 8 MB base64 limit")
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("Image exceeds 16 megapixels")
                    images[name] = image.convert("RGB")
            with lock:
                actions = policy.predict(
                    images=images,
                    instruction=observation.instruction,
                    state=observation.state,
                    graph=observation.graph,
                    seed=observation.seed,
                )
        except (
            ValueError,
            KeyError,
            TypeError,
            FloatingPointError,
            UnidentifiedImageError,
            binascii.Error,
        ) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "actions": actions.tolist(),
            "execute_steps": policy.model.config.action.execute_steps,
            "action_units": "training dataset units",
        }

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy = WISEPolicy.from_checkpoint(args.checkpoint, args.device)
    uvicorn.run(create_app(policy), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
