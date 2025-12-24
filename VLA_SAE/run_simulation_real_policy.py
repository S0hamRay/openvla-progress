import torch
import pybullet as p
import numpy as np
import cv2
from transformers import AutoTokenizer, AutoProcessor

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from simulation_environment import PyBulletManipulationEnv
import torchvision.transforms.functional as TF

def preprocess_image(rgb):
    # rgb: HWC uint8 (480x640)
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    # Pad + resize to (512, 512)
    rgb = TF.resize(rgb, policy.config.resize_imgs_with_padding)

    return rgb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



CHECKPOINT_DIR = "outputs/train/2025-12-24/01-17-20_smolvla/checkpoints/000100/pretrained_model"

policy = SmolVLAPolicy.from_pretrained(
    CHECKPOINT_DIR,
    device=DEVICE,
)

tokenizer = AutoTokenizer.from_pretrained(
    policy.config.vlm_model_name, 
    use_fast=True
)
policy.config.empty_cameras = 1

processor = AutoProcessor.from_pretrained(
    policy.config.vlm_model_name
)


policy.eval()

print(f"policy.config: {policy.config}")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
policy = policy.to(DEVICE)  # ensure model is on device

# consider resizing images if doesn't work
def make_observation(env):
    rgb = preprocess_image(env.get_wrist_camera_image())
    rgb = rgb.unsqueeze(0).unsqueeze(1).to(DEVICE)

    lang = processor(
        text="pick up the object",
        return_tensors="pt"
    )

    obs = {
        "observation.images.wrist": rgb,
        "observation.state": torch.zeros(1, 1, 6, device=DEVICE),
        "observation.language.tokens": lang["input_ids"].to(DEVICE),
        "observation.language.attention_mask": lang["attention_mask"].to(DEVICE).bool(),

        "action": torch.zeros(
            1,
            policy.config.n_action_steps,
            policy.config.output_features["action"].shape[0],
            device=DEVICE,
        )

    }
    return obs




def apply_action(env, action, current_joints):
    # Ensure numpy
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()

    # Remove batch dim
    if action.ndim == 2 and action.shape[0] == 1:
        action = action.squeeze(0)

    # Scale delta
    delta = action * 0.05

    # Pad or truncate to match robot joints
    if len(delta) < len(current_joints):
        delta = np.pad(delta, (0, len(current_joints) - len(delta)))
    elif len(delta) > len(current_joints):
        delta = delta[:len(current_joints)]

    target = current_joints + delta

    env.step(target)
    return target



env = PyBulletManipulationEnv(gui=True)

current_joints = np.zeros(env.num_joints)

current_joints = np.zeros(env.num_joints)

for step in range(2000):
    obs = make_observation(env)

    with torch.no_grad():
        action = policy.select_action(obs)
        action = action.detach().cpu().numpy()

        # Apply action and step environment in one place
        current_joints = apply_action(env, action, current_joints)

    if env.cube_in_target():
        print("✅ Success!")
        break

env.close()

