import os
import cv2
import numpy as np
from simulation_environment import PyBulletManipulationEnv

OUT_DIR = "pybullet_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

env = PyBulletManipulationEnv(gui=False)

episode_len = 200
num_episodes = 50

for ep in range(num_episodes):
    video_path = f"{OUT_DIR}/episode_{ep:03d}.mp4"
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (128, 128),
    )

    actions = []

    joint_positions = np.zeros(env.num_joints)

    for t in range(episode_len):
        joint_positions[1] = 0.5 * np.sin(t * 0.05)
        joint_positions[2] = -0.5
        joint_positions[4] = 0.3

        obs, action = env.step(joint_positions)

        frame = obs["images"]["wrist"]
        frame_bgr = (np.clip(frame, 0, 255)).astype(np.uint8)[:, :, ::-1]
        writer.write(frame_bgr)

        actions.append(action)

    writer.release()
    np.save(f"{OUT_DIR}/episode_{ep:03d}_actions.npy", np.array(actions))

    print(f"Saved episode {ep}")

env.close()
