"""
Test script to verify environment setup and see what's happening.
Run: python scripts/test_env.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import torch
import gymnasium as gym

# Import task registration
import active_recon.tasks  # noqa
from active_recon.tasks.active_scan.env_cfg import ActiveScanEnvCfg

def main():
    # Create config with specified num_envs
    env_cfg = ActiveScanEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment
    env = gym.make("ActiveScan-UR5e-v0", cfg=env_cfg)
    env.reset()
    
    print("\n" + "="*60)
    print("ENVIRONMENT TEST")
    print("="*60)
    
    # Get the unwrapped environment
    unwrapped = env.unwrapped
    
    # Check scene objects
    print("\n[SCENE OBJECTS]")
    for name in unwrapped.scene.keys():
        print(f"  - {name}")
    
    # Check robot
    robot = unwrapped.scene["robot"]
    print(f"\n[ROBOT]")
    print(f"  - Joint names: {robot.joint_names}")
    print(f"  - Body names: {robot.body_names}")
    print(f"  - Default joint pos: {robot.data.default_joint_pos[0].tolist()}")
    
    # Check if wrist_3_link exists (camera mount)
    ee_idx = None
    try:
        ee_idx = robot.find_bodies("wrist_3_link")[0][0]
        ee_pos = robot.data.body_pos_w[0, ee_idx]
        print(f"  - wrist_3_link (camera) position: {ee_pos.tolist()}")
    except Exception as e:
        print(f"  - ERROR finding wrist_3_link: {e}")
        print(f"  - Available bodies: {robot.body_names}")
    
    # Check object position
    print(f"\n[OBJECT]")
    print(f"  - Expected position: [0.5, 0.0, 0.1]")
    
    # Run a few steps with random actions
    print("\n[RUNNING 100 STEPS WITH RANDOM CAPTURES]")
    print("-"*60)
    
    for step in range(100):
        # Random arm movement + forced capture every 10 steps
        arm_action = torch.randn(unwrapped.num_envs, 6, device=unwrapped.device) * 0.1
        
        # Force capture every 10 steps
        if step % 10 == 5:
            capture_action = torch.ones(unwrapped.num_envs, 1, device=unwrapped.device)
        else:
            capture_action = torch.zeros(unwrapped.num_envs, 1, device=unwrapped.device)
        
        action = torch.cat([arm_action, capture_action], dim=-1)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get scan state
        if hasattr(unwrapped, '_scan_state'):
            scan_state = unwrapped._scan_state
            
            if step % 10 == 6:  # Right after capture
                print(f"Step {step:3d} | Captures: {scan_state['capture_count'].tolist()} | "
                      f"Just captured: {scan_state['just_captured'].tolist()} | "
                      f"Reward: {reward[0].item():.3f}")
    
    print("-"*60)
    
    if hasattr(unwrapped, '_scan_state'):
        scan_state = unwrapped._scan_state
        print(f"\n[FINAL CAPTURE COUNTS]: {scan_state['capture_count'].tolist()}")
        print(f"[COVERAGE HISTOGRAM SUM]: {scan_state['coverage_histogram'].sum(dim=-1).tolist()}")
    
    # Calculate distance to object
    if ee_idx is not None:
        ee_pos = robot.data.body_pos_w[:, ee_idx]
        obj_pos = torch.tensor([[0.5, 0.0, 0.1]], device=unwrapped.device)
        distance = torch.norm(ee_pos - obj_pos, dim=-1)
        print(f"\n[CAMERA-OBJECT DISTANCE]: {distance.tolist()} (target: 0.4m)")
    
    print("\n" + "="*60)
    print("TEST COMPLETE - Check visualization window to see Baby Yoda!")
    print("="*60)
    
    # Keep running to visualize
    print("\nKeeping window open for 10 seconds...")
    for _ in range(300):  # ~10 seconds at 30fps
        action = torch.zeros(unwrapped.num_envs, 7, device=unwrapped.device)
        env.step(action)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
