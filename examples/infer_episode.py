"""Replay observations from an episode and print predicted actions (no hardware IO)."""

import argparse
import json

from wise_vla.data.dataset import EpisodeDataset
from wise_vla.policy import WISEPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()
    policy = WISEPolicy.from_checkpoint(args.checkpoint, device=args.device)
    dataset = EpisodeDataset(args.manifest, policy.model.config.action.horizon, split=args.split)
    last_episode = None
    for index in range(min(args.frames, len(dataset))):
        sample = dataset[index]
        if sample["episode_id"] != last_episode:
            policy.reset()
            last_episode = sample["episode_id"]
        observation = {key: sample[key] for key in ("images", "instruction", "state", "graph")}
        action = policy.act(**observation, seed=42)
        print(
            json.dumps(
                {"episode": last_episode, "timestep": sample["timestep"], "action": action.tolist()}
            )
        )


if __name__ == "__main__":
    main()
