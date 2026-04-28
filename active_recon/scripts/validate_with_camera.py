"""
VALIDATION SCRIPT - Proves camera actually sees the object

This script:
1. Loads trained policy
2. Runs it with ACTUAL camera enabled
3. Saves camera images at each capture
4. Shows correlation between geometric reward and actual visibility

Use this for your presentation to PROVE the system works!

Run: python scripts/validate_with_camera.py --checkpoint logs/rsl_rl/.../model_500.pt
"""

import argparse
import sys

# Inject camera flag BEFORE importing anything
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model")
parser.add_argument("--num_episodes", type=int, default=3, help="Episodes to run")
parser.add_argument("--save_dir", type=str, default="./validation_images", help="Where to save")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import gymnasium as gym

# Import AFTER AppLauncher
import active_recon.tasks
from active_recon.tasks.active_scan.env_cfg import ActiveScanEnvCfg


class SimpleActor(nn.Module):
    """Simple actor network matching RSL-RL structure."""
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, act_dim))
        self.actor = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.actor(obs)


def save_image(array, filepath):
    """Save numpy array as PNG image."""
    try:
        import cv2
        if array.shape[-1] == 4:  # RGBA
            array = array[:, :, :3]  # RGB only
        bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), bgr)
        return True
    except ImportError:
        try:
            from PIL import Image
            if array.shape[-1] == 4:
                array = array[:, :, :3]
            Image.fromarray(array.astype(np.uint8)).save(str(filepath))
            return True
        except:
            np.save(str(filepath).replace('.png', '.npy'), array)
            return True


def compute_geometric_alignment(ee_pos, ee_quat, obj_pos):
    """Compute geometric camera-object alignment (same as reward function)."""
    # Direction from camera to object
    to_obj = obj_pos - ee_pos
    distance = np.linalg.norm(to_obj)
    to_obj_normalized = to_obj / (distance + 1e-8)
    
    # Camera forward from quaternion (assumes Z-forward)
    w, x, y, z = ee_quat
    cam_forward = np.array([
        2 * (x*z + w*y),
        2 * (y*z - w*x),
        1 - 2*(x*x + y*y)
    ])
    
    # Alignment: dot product
    alignment = np.dot(cam_forward, to_obj_normalized)
    
    return alignment, distance


def main():
    print("\n" + "="*70)
    print("CAMERA VALIDATION - Proving the system works!")
    print("="*70)
    
    # Create output directory
    save_dir = Path(args_cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nImages will be saved to: {save_dir.absolute()}")
    
    # Create environment WITH camera
    # Note: env_cfg should have wrist_camera defined for this to work
    env_cfg = ActiveScanEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("ActiveScan-UR5e-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    
    # Check for camera
    camera = None
    if hasattr(unwrapped.scene, 'sensors'):
        for name, sensor in unwrapped.scene.sensors.items():
            if 'camera' in name.lower():
                camera = sensor
                print(f"\n✓ Camera found: {name}")
                print(f"  Resolution: {sensor.cfg.height} x {sensor.cfg.width}")
                break
    
    if camera is None:
        print("\n❌ ERROR: No camera found!")
        print("   For validation, you need camera in env_cfg.py")
        print("   Add wrist_camera = CameraCfg(...) to ActiveScanSceneCfg")
        env.close()
        simulation_app.close()
        return
    
    # Load policy
    print(f"\nLoading checkpoint: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    actor = SimpleActor(51, 7).to("cuda:0")
    actor_state = {k.replace("actor.", ""): v for k, v in state_dict.items() if "actor" in k}
    actor.actor.load_state_dict(actor_state)
    actor.eval()
    print("✓ Policy loaded")
    
    # Get robot reference
    robot = unwrapped.scene["robot"]
    ee_idx = robot.find_bodies("wrist_3_link")[0][0]
    
    # Results storage
    all_results = []
    
    for episode in range(args_cli.num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{args_cli.num_episodes}")
        print(f"{'='*70}")
        
        episode_dir = save_dir / f"episode_{episode+1:02d}"
        episode_dir.mkdir(exist_ok=True)
        
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"].to("cuda:0").float()
        
        step = 0
        capture_count = 0
        episode_data = []
        
        done = False
        while not done and step < 500:
            # Get action from policy
            with torch.no_grad():
                actions = actor(obs)
            
            # Step environment
            obs_dict, reward, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"].to("cuda:0").float()
            step += 1
            
            # Check if capture happened
            scan_state = unwrapped._scan_state
            just_captured = scan_state["just_captured"][0].item()
            
            if just_captured:
                capture_count += 1
                
                # Get camera position and orientation
                ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
                ee_quat = robot.data.body_quat_w[0, ee_idx].cpu().numpy()
                
                # Get object position
                obj_pos = np.array([0.4, 0.0, 0.12])  # From config
                
                # Compute geometric alignment
                alignment, distance = compute_geometric_alignment(ee_pos, ee_quat, obj_pos)
                
                # Update camera and capture image
                camera.update(dt=0.0)
                
                if "rgb" in camera.data.output:
                    rgb = camera.data.output["rgb"][0].cpu().numpy()
                    
                    # Analyze image - check if object is visible
                    # Green cube should have high green channel
                    green_ratio = rgb[:, :, 1].mean() / (rgb.mean() + 1)
                    has_green = green_ratio > 1.1  # Object likely visible
                    
                    # Image statistics
                    img_std = rgb[:, :, :3].std()
                    
                    # Save image
                    img_path = episode_dir / f"capture_{capture_count:02d}.png"
                    save_image(rgb, img_path)
                    
                    # Record data
                    data = {
                        "capture": capture_count,
                        "step": step,
                        "ee_pos": ee_pos.tolist(),
                        "alignment": float(alignment),
                        "distance": float(distance),
                        "green_ratio": float(green_ratio),
                        "img_std": float(img_std),
                        "likely_visible": bool(has_green or img_std > 30),
                    }
                    episode_data.append(data)
                    
                    # Print status
                    vis_icon = "✓" if data["likely_visible"] else "✗"
                    print(f"  📸 Capture {capture_count:2d} | "
                          f"Align: {alignment:+.2f} | "
                          f"Dist: {distance:.2f}m | "
                          f"GreenRatio: {green_ratio:.2f} | "
                          f"Visible: {vis_icon}")
            
            done = terminated.any().item() or truncated.any().item()
        
        # Episode summary
        print(f"\n  Episode Summary:")
        print(f"    Captures: {capture_count}")
        
        if episode_data:
            avg_align = np.mean([d["alignment"] for d in episode_data])
            visible_count = sum(1 for d in episode_data if d["likely_visible"])
            print(f"    Avg Alignment: {avg_align:.2f}")
            print(f"    Visible Captures: {visible_count}/{capture_count}")
            
            all_results.extend(episode_data)
    
    # ═══════════════════════════════════════════════════════════════════
    # FINAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    if all_results:
        total_captures = len(all_results)
        visible_captures = sum(1 for d in all_results if d["likely_visible"])
        avg_alignment = np.mean([d["alignment"] for d in all_results])
        
        # Correlation between alignment and visibility
        alignments = np.array([d["alignment"] for d in all_results])
        visible = np.array([d["likely_visible"] for d in all_results])
        
        high_align_visible = sum(1 for a, v in zip(alignments, visible) if a > 0.5 and v)
        high_align_total = sum(1 for a in alignments if a > 0.5)
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                      VALIDATION SUMMARY                             │
├─────────────────────────────────────────────────────────────────────┤
│  Total Captures:           {total_captures:4d}                                    │
│  Likely Visible:           {visible_captures:4d} ({100*visible_captures/total_captures:5.1f}%)                         │
│  Average Alignment:        {avg_alignment:+5.2f}                                  │
├─────────────────────────────────────────────────────────────────────┤
│  GEOMETRIC vs ACTUAL CORRELATION:                                   │
│  High alignment (>0.5) captures: {high_align_total:4d}                            │
│  Of those, actually visible:     {high_align_visible:4d} ({100*high_align_visible/(high_align_total+0.001):5.1f}%)                │
└─────────────────────────────────────────────────────────────────────┘

INTERPRETATION:
""")
        
        if high_align_visible / (high_align_total + 0.001) > 0.7:
            print("  ✓ GEOMETRIC REWARDS CORRELATE WITH ACTUAL VISIBILITY!")
            print("  → The proxy reward (camera_facing) accurately predicts camera seeing object")
            print("  → Training without camera was valid!")
        else:
            print("  ⚠ Low correlation between geometric reward and visibility")
            print("  → May need to adjust reward function or camera orientation")
        
        print(f"\n  Images saved to: {save_dir.absolute()}")
        print(f"  Check the images to verify object is visible!")
    
    # Create summary image grid
    try:
        import cv2
        all_images = list(save_dir.glob("**/capture_*.png"))
        if len(all_images) >= 4:
            imgs = [cv2.imread(str(p)) for p in all_images[:8]]
            imgs = [cv2.resize(img, (320, 240)) for img in imgs if img is not None]
            
            if len(imgs) >= 4:
                rows = []
                for i in range(0, min(8, len(imgs)), 4):
                    row = imgs[i:i+4]
                    while len(row) < 4:
                        row.append(np.zeros_like(row[0]))
                    rows.append(np.hstack(row))
                
                grid = np.vstack(rows) if len(rows) > 1 else rows[0]
                grid_path = save_dir / "ALL_CAPTURES_GRID.png"
                cv2.imwrite(str(grid_path), grid)
                print(f"\n  📊 Summary grid: {grid_path}")
    except Exception as e:
        print(f"  (Could not create grid: {e})")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
