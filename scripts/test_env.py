"""
Test script for Volume Reconstruction environment.

This script:
1. Creates the environment
2. Runs random actions
3. Displays coverage statistics
4. Visualizes the volume bounds

Usage:
    python scripts/test_env.py --num_envs 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Volume Reconstruction environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_steps", type=int, default=200, help="Steps to run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

# Import task registration
import volume_recon.tasks  # noqa: F401
from volume_recon.tasks.volume_scan.env_cfg import VolumeReconEnvCfg, VOLUME_BOUNDS, VOXEL_RESOLUTION


def main():
    # Create environment
    env_cfg = VolumeReconEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    env = gym.make("VolumeRecon-UR5e-v0", cfg=env_cfg)
    obs, _ = env.reset()
    
    unwrapped = env.unwrapped
    
    print("\n" + "="*70)
    print("VOLUME RECONSTRUCTION ENVIRONMENT TEST")
    print("="*70)
    
    # Volume info
    print("\n[VOLUME CONFIGURATION]")
    print(f"  Bounds X: [{VOLUME_BOUNDS['x_min']:.2f}, {VOLUME_BOUNDS['x_max']:.2f}] m")
    print(f"  Bounds Y: [{VOLUME_BOUNDS['y_min']:.2f}, {VOLUME_BOUNDS['y_max']:.2f}] m")
    print(f"  Bounds Z: [{VOLUME_BOUNDS['z_min']:.2f}, {VOLUME_BOUNDS['z_max']:.2f}] m")
    print(f"  Resolution: {VOXEL_RESOLUTION}")
    print(f"  Total voxels: {VOXEL_RESOLUTION[0] * VOXEL_RESOLUTION[1] * VOXEL_RESOLUTION[2]}")
    
    # Scene info
    print("\n[SCENE OBJECTS]")
    for name in unwrapped.scene.keys():
        print(f"  - {name}")
    
    # Robot info
    robot = unwrapped.scene["robot"]
    print(f"\n[ROBOT]")
    print(f"  Joint names: {robot.joint_names}")
    print(f"  Body names: {robot.body_names}")
    
    # Camera position
    camera_ids = robot.find_bodies("wrist_3_link")[0]
    camera_pos = robot.data.body_pos_w[0, camera_ids[0]]
    print(f"  Camera position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
    
    # Run test
    print("\n[RUNNING TEST WITH RANDOM CAPTURES]")
    print("-"*70)
    
    for step in range(args_cli.num_steps):
        # Random arm movement
        arm_action = torch.randn(unwrapped.num_envs, 6, device=unwrapped.device) * 0.3
        
        # Capture every 20 steps
        if step % 20 == 10:
            capture_action = torch.ones(unwrapped.num_envs, 1, device=unwrapped.device)
        else:
            capture_action = torch.zeros(unwrapped.num_envs, 1, device=unwrapped.device)
        
        action = torch.cat([arm_action, capture_action], dim=-1)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get tracker stats
        if hasattr(unwrapped, '_volume_tracker'):
            tracker = unwrapped._volume_tracker
            
            if step % 20 == 11:  # Right after capture
                coverage = tracker.get_coverage_percentage()
                captures = tracker.capture_count
                multi_view = tracker.get_multi_view_score()
                new_voxels = tracker.new_voxels_this_capture
                
                print(f"Step {step:3d} | "
                      f"Captures: {captures[0].item():2d} | "
                      f"Coverage: {coverage[0].item()*100:5.1f}% | "
                      f"Multi-view: {multi_view[0].item():.2f} | "
                      f"New voxels: {new_voxels[0].item():.0f} | "
                      f"Reward: {reward[0].item():+.3f}")
    
    print("-"*70)
    
    # Final stats
    if hasattr(unwrapped, '_volume_tracker'):
        tracker = unwrapped._volume_tracker
        print(f"\n[FINAL STATISTICS]")
        print(f"  Total captures: {tracker.capture_count[0].item()}")
        print(f"  Coverage: {tracker.get_coverage_percentage()[0].item()*100:.1f}%")
        print(f"  Multi-view score: {tracker.get_multi_view_score()[0].item():.2f}")
        print(f"  Voxels seen: {(tracker.coverage_counts[0] > 0).sum().item()} / {tracker.num_voxels}")
    
    # Camera final position
    camera_pos = robot.data.body_pos_w[0, camera_ids[0]]
    print(f"\n[FINAL CAMERA POSITION]: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
    
    # Check if camera is in volume
    in_volume = (
        VOLUME_BOUNDS['x_min'] <= camera_pos[0] <= VOLUME_BOUNDS['x_max'] and
        VOLUME_BOUNDS['y_min'] <= camera_pos[1] <= VOLUME_BOUNDS['y_max'] and
        VOLUME_BOUNDS['z_min'] <= camera_pos[2] <= VOLUME_BOUNDS['z_max']
    )
    print(f"  Camera in volume: {in_volume}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    # Keep visualization open
    print("\nKeeping window open for 10 seconds...")
    print("Look for the robot arm in the visualization.")
    print("The volume to reconstruct is the space in front of the robot.")
    
    for _ in range(300):
        action = torch.zeros(unwrapped.num_envs, 7, device=unwrapped.device)
        env.step(action)
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
