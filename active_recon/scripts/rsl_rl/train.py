"""
Training Script with Image Capture Every 10 Iterations

USAGE:
python scripts/rsl_rl/train.py --task ActiveScan-UR5e-v0 --num_envs 32 --max_iterations 500 --enable_cameras

Images saved to: logs/rsl_rl/[task]/[timestamp]/debug_images/
"""

from __future__ import annotations

import argparse
import sys
import os

# Add enable_cameras by default for image capture
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train with image capture")
parser.add_argument("--task", type=str, default="ActiveScan-UR5e-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--image_interval", type=int, default=10, help="Save image every N iterations")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after AppLauncher
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import active_recon.tasks  # noqa: F401
from active_recon.tasks.active_scan.env_cfg import ActiveScanEnvCfg


# ═══════════════════════════════════════════════════════════════════════════════
# PPO CONFIG - DEFINED INLINE
# ═══════════════════════════════════════════════════════════════════════════════

@configclass
class ActiveScanPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO with AGGRESSIVE EXPLORATION - fixes -0.3 stuck reward."""
    
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 100
    experiment_name = "ActiveScan-UR5e"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "isaaclab"
    wandb_project = "isaaclab"
    resume = False
    load_run = None
    load_checkpoint = None
    empirical_normalization = False
    
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=2.0,                    # MORE random initial actions
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.1,                      # 10x original - MUCH more exploration!
        learning_rate=3e-4,                    # Faster learning
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        max_grad_norm=1.0,
        desired_kl=0.02,                       # Allow bigger policy updates
        schedule="adaptive",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE CAPTURE GLOBALS
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_COUNTER = {"step": 0, "last_capture": -1}
IMAGE_DIR = None
IMAGE_INTERVAL = 10


def save_camera_image(env, iteration: int):
    """Capture and save image from camera."""
    global IMAGE_DIR
    
    if IMAGE_DIR is None:
        return
        
    try:
        # Navigate through wrappers to get Isaac Lab env
        unwrapped = env
        while hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, 'unwrapped'):
            unwrapped = unwrapped.unwrapped
        
        # Find camera in sensors
        camera = None
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'sensors'):
            for name, sensor in unwrapped.scene.sensors.items():
                if 'camera' in name.lower():
                    camera = sensor
                    break
        
        if camera is None:
            return
        
        # Update camera to get fresh data
        camera.update(dt=0.0)
        
        if "rgb" not in camera.data.output:
            return
        
        # Get image from env 0
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        
        # Convert RGBA to RGB if needed
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        filepath = IMAGE_DIR / f"iter_{iteration:05d}.png"
        
        try:
            import cv2
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), bgr)
        except ImportError:
            from PIL import Image
            Image.fromarray(rgb.astype(np.uint8)).save(str(filepath))
        
        # Print debug info
        robot = unwrapped.scene["robot"]
        ee_idx = robot.find_bodies("wrist_3_link")[0][0]
        ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
        
        print(f"[IMAGE iter {iteration:4d}] Saved: {filepath.name} | "
              f"Camera: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]")
        
    except Exception as e:
        print(f"[IMAGE] Error at iter {iteration}: {e}")


def create_step_wrapper(original_step, env):
    """Create a wrapper around env.step that captures images."""
    def wrapped_step(actions):
        global IMAGE_COUNTER, IMAGE_INTERVAL
        
        result = original_step(actions)
        
        # Capture image every N iterations (approximately)
        IMAGE_COUNTER["step"] += 1
        iteration = IMAGE_COUNTER["step"] // 24  # Approximate iteration from steps
        
        if iteration > IMAGE_COUNTER["last_capture"] and iteration % IMAGE_INTERVAL == 0:
            IMAGE_COUNTER["last_capture"] = iteration
            save_camera_image(env, iteration)
        
        return result
    
    return wrapped_step


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main training function."""
    global IMAGE_DIR, IMAGE_INTERVAL
    
    # Setup environment
    env_cfg = ActiveScanEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Wrap with RSL-RL wrapper
    env = RslRlVecEnvWrapper(env)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/rsl_rl/{args_cli.task}/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup image capture
    IMAGE_DIR = log_dir / "debug_images"
    IMAGE_INTERVAL = args_cli.image_interval
    print(f"[IMAGE] Will save debug images to: {IMAGE_DIR}")
    
    # Monkey-patch the step function to capture images
    original_step = env.step
    env.step = create_step_wrapper(original_step, env)
    
    # Setup agent
    agent_cfg = ActiveScanPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.save_interval = args_cli.save_interval
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=env.device)
    
    print(f"\n{'='*70}")
    print(f"[INFO] Starting training WITH IMAGE CAPTURE...")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Max iterations: {args_cli.max_iterations}")
    print(f"[INFO] Image capture every: {args_cli.image_interval} iterations")
    print(f"[INFO] Log dir: {log_dir}")
    print(f"{'='*70}\n")
    
    # Capture initial image
    save_camera_image(env, 0)
    
    # Run training using standard runner.learn()
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    
    print(f"\n{'='*70}")
    print(f"[INFO] Training complete!")
    print(f"[INFO] Debug images: {IMAGE_DIR}")
    print(f"[INFO] Final model: {log_dir / 'model_final.pt'}")
    print(f"{'='*70}\n")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()