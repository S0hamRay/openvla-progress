import os
import numpy as np
import pandas as pd

ROOT = "lerobot_pybullet"
os.makedirs(f"{ROOT}/videos/observation.images.wrist", exist_ok=True)
os.makedirs(f"{ROOT}/data", exist_ok=True)

rows = []

for ep in range(50):
    os.rename(
        f"pybullet_dataset/episode_{ep:03d}.mp4",
        f"{ROOT}/videos/observation.images.wrist/episode_{ep:03d}.mp4"
    )

    actions = np.load(f"pybullet_dataset/episode_{ep:03d}_actions.npy")

    for t, act in enumerate(actions):
        rows.append({
            "episode_index": ep,
            "timestamp": t,
            "action": act.tolist()
        })

df = pd.DataFrame(rows)
df.to_parquet(f"{ROOT}/data/actions.parquet")
