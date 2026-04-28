"""
Capture test images from wrist camera to verify setup.
This will save actual images so you can SEE what the camera sees.

Run: python scripts/capture_test_images.py --save_dir ./test_captures
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./test_captures", help="Where to save images")
parser.add_argument("--num_images", type=int, default=8, help="Number of test images")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from pathlib import Path
import gymnasium as gym

import active_recon.tasks
from active_recon.tasks.active_scan.env_cfg import ActiveScanEnvCfg


def save_image(image_array, filepath):
    """Save numpy array as image."""
    try:
        import cv2
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
            bgr = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), bgr)
            return True
    except ImportError:
        # Fallback to PIL
        try:
            from PIL import Image
            img = Image.fromarray(image_array[:, :, :3])
            img.save(str(filepath))
            return True
        except ImportError:
            # Fallback to raw numpy save
            np.save(str(filepath).replace('.png', '.npy'), image_array)
            return True
    return False


def main():
    print("\n" + "="*70)
    print("CAMERA IMAGE CAPTURE TEST")
    print("="*70)
    
    # Create save directory
    save_dir = Path(args_cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SAVE DIR] {save_dir.absolute()}")
    
    # Create environment
    env_cfg = ActiveScanEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("ActiveScan-UR5e-v0", cfg=env_cfg)
    obs, _ = env.reset()
    
    unwrapped = env.unwrapped
    
    # Check for camera
    print("\n[CAMERA CHECK]")
    camera = None
    if hasattr(unwrapped.scene, 'sensors'):
        for name, sensor in unwrapped.scene.sensors.items():
            print(f"  Found sensor: {name}")
            if 'camera' in name.lower():
                camera = sensor
                print(f"    ✓ This is our camera!")
    
    if camera is None:
        print("\n  ❌ NO CAMERA FOUND!")
        print("  You need to add CameraCfg to your env_cfg.py")
        print("  See: ENV_CFG_CAMERA_UPDATE.md for instructions")
        env.close()
        simulation_app.close()
        return
    
    # Get robot
    robot = unwrapped.scene["robot"]
    ee_idx = robot.find_bodies("wrist_3_link")[0][0]
    
    print(f"\n[CAPTURING {args_cli.num_images} TEST IMAGES]")
    print("-"*50)
    
    captured = 0
    step = 0
    
    # Define some test joint positions for different viewpoints
    test_positions = [
        {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.0, "elbow_joint": 1.2, 
         "wrist_1_joint": -1.5, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        {"shoulder_pan_joint": 0.5, "shoulder_lift_joint": -1.2, "elbow_joint": 1.5,
         "wrist_1_joint": -1.3, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        {"shoulder_pan_joint": -0.5, "shoulder_lift_joint": -0.8, "elbow_joint": 1.0,
         "wrist_1_joint": -1.8, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        {"shoulder_pan_joint": 0.8, "shoulder_lift_joint": -1.4, "elbow_joint": 1.8,
         "wrist_1_joint": -1.0, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        {"shoulder_pan_joint": -0.8, "shoulder_lift_joint": -1.0, "elbow_joint": 1.3,
         "wrist_1_joint": -1.6, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        {"shoulder_pan_joint": 0.3, "shoulder_lift_joint": -0.6, "elbow_joint": 0.8,
         "wrist_1_joint": -2.0, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        {"shoulder_pan_joint": -0.3, "shoulder_lift_joint": -1.5, "elbow_joint": 2.0,
         "wrist_1_joint": -0.8, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
        {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -0.5, "elbow_joint": 0.5,
         "wrist_1_joint": -2.2, "wrist_2_joint": -1.57, "wrist_3_joint": 0.0},
    ]
    
    while captured < args_cli.num_images and simulation_app.is_running():
        # Apply joint positions (move arm to test position)
        if captured < len(test_positions):
            target_pos = test_positions[captured]
            # Convert to action (simplified - just move toward target)
            current_joints = robot.data.joint_pos[0].cpu().numpy()
            joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
            
            action = torch.zeros(1, 7, device=unwrapped.device)
            for i, name in enumerate(joint_names):
                if name in target_pos:
                    action[0, i] = (target_pos[name] - current_joints[i]) * 0.5
        else:
            # Random action
            action = torch.randn(1, 7, device=unwrapped.device) * 0.1
        
        action[0, 6] = -1.0  # Don't trigger capture action
        
        obs, _, _, _, _ = env.step(action)
        step += 1
        
        # Every 50 steps, capture an image
        if step % 50 == 0:
            # Update camera (render)
            camera.update(dt=0.0)
            
            # Get camera data
            camera_data = camera.data
            
            if "rgb" in camera_data.output:
                rgb_tensor = camera_data.output["rgb"][0]  # (H, W, 4) RGBA
                rgb_np = rgb_tensor[:, :, :3].cpu().numpy().astype(np.uint8)
                
                # Get camera position
                ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
                
                # Save image
                img_path = save_dir / f"capture_{captured:02d}.png"
                if save_image(rgb_np, img_path):
                    print(f"  📸 Image {captured+1}: {img_path.name}")
                    print(f"      Camera pos: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]")
                    print(f"      Image shape: {rgb_np.shape}")
                    print(f"      Pixel range: [{rgb_np.min()}, {rgb_np.max()}]")
                    
                    # Check if image is not all black/uniform
                    std = rgb_np.std()
                    if std < 5:
                        print(f"      ⚠️ Image appears uniform (std={std:.1f}) - object might not be visible!")
                    else:
                        print(f"      ✓ Image has content (std={std:.1f})")
                    
                    captured += 1
            else:
                print(f"  ❌ No RGB data available from camera!")
                print(f"     Available: {list(camera_data.output.keys())}")
                break
    
    # Create a grid of all captures
    print(f"\n[CREATING CAPTURE GRID]")
    try:
        import cv2
        images = []
        for i in range(captured):
            img_path = save_dir / f"capture_{i:02d}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (320, 240))  # Smaller for grid
                images.append(img)
        
        if len(images) >= 4:
            # Create 2xN grid
            rows = []
            for i in range(0, len(images), 4):
                row_imgs = images[i:i+4]
                while len(row_imgs) < 4:
                    row_imgs.append(np.zeros_like(row_imgs[0]))
                rows.append(np.hstack(row_imgs))
            
            grid = np.vstack(rows)
            grid_path = save_dir / "capture_grid.png"
            cv2.imwrite(str(grid_path), grid)
            print(f"  ✓ Saved grid: {grid_path}")
    except ImportError:
        print("  (cv2 not available, skipping grid)")
    
    # Summary
    print("\n" + "="*70)
    print("CAPTURE TEST COMPLETE")
    print("="*70)
    print(f"""
Images saved to: {save_dir.absolute()}

Check the images:
  - If you see Baby Yoda/object → Camera works! ✓
  - If images are black → Lighting issue
  - If images show only ground/sky → Camera pointing wrong direction
  - If object is tiny/huge → Scale issue

Next steps:
  1. Open {save_dir}/capture_grid.png to verify camera sees object
  2. If OK, proceed with training
  3. If not, check object scale and camera orientation
""")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()