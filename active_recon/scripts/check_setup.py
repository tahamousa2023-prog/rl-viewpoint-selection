"""
Quick validation script - run BEFORE training to verify setup.
Run: python scripts/check_setup.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import active_recon.tasks
from active_recon.tasks.active_scan.env_cfg import ActiveScanEnvCfg


def main():
    print("\n" + "="*70)
    print("SETUP VALIDATION CHECK")
    print("="*70)
    
    # Create environment
    env_cfg = ActiveScanEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("ActiveScan-UR5e-v0", cfg=env_cfg)
    obs, _ = env.reset()
    
    unwrapped = env.unwrapped
    
    # ═══════════════════════════════════════════════════════════════════
    # CHECK 1: Camera exists?
    # ═══════════════════════════════════════════════════════════════════
    print("\n[CHECK 1] Camera Sensor")
    print("-" * 50)
    
    camera = None
    camera_found = False
    
    # Check in scene.sensors dictionary
    if hasattr(unwrapped.scene, 'sensors') and unwrapped.scene.sensors:
        for name, sensor in unwrapped.scene.sensors.items():
            if 'camera' in name.lower():
                camera = sensor
                camera_found = True
                print(f"  ✓ Camera found: {name}")
                print(f"    Resolution: {sensor.cfg.height} x {sensor.cfg.width}")
                print(f"    Data types: {sensor.cfg.data_types}")
                break
        if not camera_found:
            print(f"  Available sensors: {list(unwrapped.scene.sensors.keys())}")
    
    # Also check if wrist_camera exists as attribute
    if not camera_found and hasattr(unwrapped.scene, 'wrist_camera'):
        camera = unwrapped.scene.wrist_camera
        camera_found = True
        print(f"  ✓ Camera found: wrist_camera (as attribute)")
        print(f"    Resolution: {camera.cfg.height} x {camera.cfg.width}")
        print(f"    Data types: {camera.cfg.data_types}")
    
    if not camera_found:
        print(f"  ✗ Camera NOT found in scene!")
        print(f"    Available sensors: {list(unwrapped.scene.sensors.keys()) if hasattr(unwrapped.scene, 'sensors') else 'None'}")
        print(f"")
        print(f"    To add camera, update env_cfg.py with:")
        print(f"    1. Add import: from isaaclab.sensors import CameraCfg")
        print(f"    2. Add wrist_camera config to ActiveScanSceneCfg")
        print(f"    See: ENV_CFG_CAMERA_UPDATE.md for complete code")
    
    # ═══════════════════════════════════════════════════════════════════
    # CHECK 2: Robot exists and can move?
    # ═══════════════════════════════════════════════════════════════════
    print("\n[CHECK 2] Robot Articulation")
    print("-" * 50)
    
    if "robot" in unwrapped.scene.articulations:
        robot = unwrapped.scene["robot"]
        print(f"  ✓ Robot found")
        print(f"    Joint count: {robot.num_joints}")
        print(f"    Body count: {robot.num_bodies}")
        
        # Find end-effector
        try:
            ee_idx = robot.find_bodies("wrist_3_link")[0][0]
            ee_pos = robot.data.body_pos_w[0, ee_idx]
            print(f"    End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        except:
            print(f"  ✗ Could not find wrist_3_link!")
    else:
        print(f"  ✗ Robot NOT found!")
    
    # ═══════════════════════════════════════════════════════════════════
    # CHECK 3: Object exists?
    # ═══════════════════════════════════════════════════════════════════
    print("\n[CHECK 3] Target Object")
    print("-" * 50)
    
    # Check via USD stage
    stage = unwrapped.sim.stage
    object_prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
    if object_prim.IsValid():
        print(f"  ✓ Object found at /World/envs/env_0/Object")
        
        from pxr import UsdGeom
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(object_prim)
        bbox_range = bbox.GetRange()
        size = bbox_range.GetSize()
        print(f"    Bounding box size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}] m")
    else:
        print(f"  ✗ Object NOT found!")
    
    # ═══════════════════════════════════════════════════════════════════
    # CHECK 4: Capture mechanism works?
    # ═══════════════════════════════════════════════════════════════════
    print("\n[CHECK 4] Capture Mechanism")
    print("-" * 50)
    
    # Try triggering a capture
    action = torch.zeros(1, 7, device=unwrapped.device)
    action[0, 6] = 1.0  # Capture action = positive
    
    obs, reward, _, _, _ = env.step(action)
    
    scan_state = unwrapped._scan_state
    captures = scan_state["capture_count"][0].item()
    just_captured = scan_state["just_captured"][0].item()
    
    if just_captured or captures > 0:
        print(f"  ✓ Capture triggered successfully!")
        print(f"    Captures so far: {captures}")
    else:
        print(f"  ✗ Capture did NOT trigger!")
        print(f"    Check BinaryAction threshold in mdp/__init__.py")
    
    # ═══════════════════════════════════════════════════════════════════
    # CHECK 5: Rewards are computed?
    # ═══════════════════════════════════════════════════════════════════
    print("\n[CHECK 5] Reward Computation")
    print("-" * 50)
    
    print(f"  Step reward: {reward[0].item():.4f}")
    
    # Check reward manager
    if hasattr(unwrapped, 'reward_manager'):
        print(f"  Active reward terms:")
        for term_name in unwrapped.reward_manager.active_terms:
            print(f"    • {term_name}")
    
    # ═══════════════════════════════════════════════════════════════════
    # CHECK 6: Observation space
    # ═══════════════════════════════════════════════════════════════════
    print("\n[CHECK 6] Observation Space")
    print("-" * 50)
    
    if isinstance(obs, dict):
        obs_tensor = obs["policy"]
    else:
        obs_tensor = obs
    print(f"  Observation shape: {obs_tensor.shape}")
    print(f"  Expected: (1, 51) = joint_pos(6) + joint_vel(6) + ee_pos(3) + obj_pos(3) + captures(1) + histogram(32)")
    
    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
If all checks pass (✓), you're ready to train!

If camera check fails:
  → Training still works with GEOMETRIC rewards (no images needed)
  → Camera is only needed for actual image capture (evaluation phase)

If capture check fails:
  → Check BinaryAction._capture_threshold (should be 0.0, not 0.5)
  → Capture action > 0 should trigger capture

Run training:
  python scripts/rsl_rl/train.py --task ActiveScan-UR5e-v0 --num_envs 32 --max_iterations 500
""")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()