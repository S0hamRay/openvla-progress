import numpy as np
from simulation_environment import PyBulletManipulationEnv

env = PyBulletManipulationEnv(gui=True)

# Neutral pose
joint_positions = [0.0] * env.num_joints

for step in range(2000):
    # Simple scripted motion
    joint_positions[1] = 0.5 * np.sin(step * 0.01)
    joint_positions[2] = -0.5
    joint_positions[4] = 0.3

    env.step(joint_positions)

    if step % 100 == 0:
        rgb = env.get_wrist_camera_image()
        print(f"Step {step}, camera image shape: {rgb.shape}")

    if env.cube_in_target():
        print("✅ Cube reached target!")
        break

env.close()
