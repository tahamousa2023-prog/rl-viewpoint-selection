"""
scripts/rsl_rl/play.py

Play/evaluate a trained Volume Reconstruction agent.
Saves captured camera poses to JSON and RGB images to disk for VGGT input.

Observation vector layout (from ObservationsCfg in env_cfg.py):
    [0:6]   joint_pos         (6 DOF)
    [6:12]  joint_vel         (6 DOF)
    [12:15] camera_pos        (x, y, z)
    [15:19] camera_quat       (w, x, y, z)
    [19]    coverage_pct
    [20]    capture_count
    [21]    multi_view_score
    [22:86] coverage_map      (64-dim downsampled)

Note: obs returned by RslRlVecEnvWrapper is a TensorDict.
      Camera pose is read via obs["policy"][0, 12:15] / [0, 15:19].

Output structure when --save-poses and --save-images are both set:
    <poses-dir>/
        episode_01_poses.json            ← pose list for VGGT
        episode_01_images/
            capture_01.png               ← RGB frame at each capture
            capture_02.png
            ...

Usage:
    # Watch the agent play:
    python scripts/rsl_rl/play.py \
        --task VolumeRecon-UR5e-v0 \
        --checkpoint logs/taha/exp_07_stricter_wasteful/2026-03-22_15-56-14/model_final.pt \
        --num_episodes 3

    # Headless + save poses + save images for VGGT:
    python scripts/rsl_rl/play.py \
        --task VolumeRecon-UR5e-v0 \
        --checkpoint logs/taha/exp_07_stricter_wasteful/2026-03-22_15-56-14/model_final.pt \
        --num_episodes 1 \
        --save-poses \
        --save-images \
        --poses-dir outputs/poses/exp_07 \
        --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../agents"))

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Play trained Volume Reconstruction agent")
parser.add_argument("--task",         type=str, default="VolumeRecon-UR5e-v0")
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--checkpoint",   type=str, required=True,
                    help="Path to .pt checkpoint file")
parser.add_argument("--num_episodes", type=int, default=3,
                    help="Number of evaluation episodes to run")
parser.add_argument("--save-poses",   action="store_true",
                    help="Save camera poses at each capture to JSON for VGGT input")
parser.add_argument("--save-images",  action="store_true",
                    help="Save rendered RGB image at each capture (requires CameraCfg in env_cfg)")
parser.add_argument("--poses-dir",    type=str, default="outputs/poses",
                    help="Directory to save pose JSON files and image folders")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (must come after AppLauncher) ─────────────────────────────────────
import torch
import numpy as np
from PIL import Image
import gymnasium as gym

try:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner, RslRlVecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner
        from isaaclab_rl.rsl_rl.vecenv import RslRlVecEnvWrapper
    except ImportError:
        from rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import volume_recon.tasks  # noqa: F401 — registers VolumeRecon-UR5e-v0
from volume_recon.tasks.volume_scan.env_cfg import VolumeReconEnvCfg
from rsl_rl_ppo_cfg import VolumeReconPPORunnerCfg

# Observation vector indices for camera pose inside obs["policy"]
# Layout: joint_pos(6) + joint_vel(6) + camera_pos(3) + camera_quat(4) + ...
OBS_CAM_POS_SLICE  = slice(12, 15)   # [x, y, z]
OBS_CAM_QUAT_SLICE = slice(15, 19)   # [w, x, y, z]


def save_rgb_image(env, episode: int, capture_idx: int, poses_dir: str) -> str | None:
    """
    Save the current RGB frame from the wrist camera to disk.

    Returns the relative path to the saved image, or None if saving failed.

    The camera sensor is accessed via env.unwrapped.scene["wrist_camera"].
    Data shape: (num_envs, H, W, 4) — RGBA uint8.
    We take env 0, drop the alpha channel, and save as PNG.
    """
    try:
        camera = env.unwrapped.scene["wrist_camera"]
        # data.output["rgb"] shape: (num_envs, H, W, 4) — RGBA, uint8
        rgba = camera.data.output["rgb"][0]  # (H, W, 4) on GPU
        rgb  = rgba[:, :, :3].cpu().numpy().astype(np.uint8)  # (H, W, 3)

        img_dir = Path(poses_dir) / f"episode_{episode:02d}_images"
        img_dir.mkdir(parents=True, exist_ok=True)

        img_path = img_dir / f"capture_{capture_idx:02d}.png"
        Image.fromarray(rgb).save(img_path)

        return str(img_path)

    except KeyError:
        # wrist_camera not in scene — env_cfg.py does not have CameraCfg yet
        print("  [WARNING] wrist_camera not found in scene. "
              "Add CameraCfg to VolumeReconSceneCfg in env_cfg.py to enable image saving.")
        return None

    except Exception as e:
        print(f"  [WARNING] Image save failed: {e}")
        return None


def main():

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = VolumeReconEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_cfg = VolumeReconPPORunnerCfg()
    runner    = RslRlOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")

    print(f"\nLoading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    # ── Output directory ──────────────────────────────────────────────────────
    if args_cli.save_poses or args_cli.save_images:
        Path(args_cli.poses_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory   → {args_cli.poses_dir}/")
        print(f"Save poses         : {'YES' if args_cli.save_poses  else 'NO'}")
        print(f"Save images        : {'YES' if args_cli.save_images else 'NO'}")

    print("\n" + "=" * 70)
    print("PLAYING TRAINED VOLUME RECONSTRUCTION AGENT")
    print("=" * 70)

    # ── Episode loop ──────────────────────────────────────────────────────────
    for episode in range(args_cli.num_episodes):
        print(f"\n--- Episode {episode + 1}/{args_cli.num_episodes} ---")

        obs, _ = env.reset()

        done         = torch.zeros(args_cli.num_envs, dtype=torch.bool, device="cuda:0")
        total_reward = torch.zeros(args_cli.num_envs, device="cuda:0")
        step         = 0

        # Running stats (updated from tracker before each step)
        last_coverage    = torch.zeros(args_cli.num_envs, device="cuda:0")
        last_captures    = torch.zeros(args_cli.num_envs, device="cuda:0")
        last_multiview   = torch.zeros(args_cli.num_envs, device="cuda:0")
        last_voxels_seen = 0

        captured_poses = []  # [{step, capture, coverage, pos, quat, image_path}, ...]

        while not done.all():

            # ── Snapshot tracker BEFORE step ──────────────────────────────────
            # Must read before step() because a terminal reset inside step()
            # will zero capture_count before we can detect the capture that fired.
            tracker      = env.unwrapped._volume_tracker
            pre_captures = tracker.capture_count.float().clone()

            last_coverage    = tracker.get_coverage_percentage().clone()
            last_captures    = pre_captures
            last_multiview   = tracker.get_multi_view_score().clone()
            last_voxels_seen = int((tracker.coverage_counts[0] > 0).sum().item())

            # ── Policy step ───────────────────────────────────────────────────
            with torch.no_grad():
                actions = policy(obs)

            obs, reward, dones, info = env.step(actions)
            done          = dones
            total_reward += reward
            step         += 1

            # ── Detect capture fires ──────────────────────────────────────────
            # capture_count increases by 1 each time the agent fires a capture.
            # obs is the POST-step observation — camera pose reflects the
            # position at which the capture was triggered.
            # obs is a TensorDict — unwrap with obs["policy"] to get flat tensor.
            post_captures = tracker.capture_count.float()

            if post_captures[0] > pre_captures[0]:
                capture_idx = int(post_captures[0].item())

                # ── Camera pose from obs ───────────────────────────────────────
                obs_tensor = obs["policy"]  # (num_envs, obs_dim) plain tensor
                cam_pos  = obs_tensor[0, OBS_CAM_POS_SLICE].cpu().numpy()
                cam_quat = obs_tensor[0, OBS_CAM_QUAT_SLICE].cpu().numpy()

                # ── Optionally save RGB image ──────────────────────────────────
                img_path = None
                if args_cli.save_images:
                    img_path = save_rgb_image(
                        env, episode + 1, capture_idx, args_cli.poses_dir
                    )

                # ── Record this capture ────────────────────────────────────────
                if args_cli.save_poses:
                    entry = {
                        "step":     step,
                        "capture":  capture_idx,
                        "coverage": float(last_coverage[0].item()),
                        "pos":      cam_pos.tolist(),    # [x, y, z]
                        "quat":     cam_quat.tolist(),   # [w, x, y, z]
                    }
                    if img_path is not None:
                        entry["image"] = img_path
                    captured_poses.append(entry)

                print(
                    f"  [CAPTURE {capture_idx:2d}] "
                    f"step={step:4d}  "
                    f"coverage={last_coverage[0].item() * 100:5.1f}%  "
                    f"pos={[round(v, 3) for v in cam_pos.tolist()]}"
                    + (f"  img saved" if img_path else "")
                )

            # ── Periodic progress print ───────────────────────────────────────
            if step % 50 == 0:
                print(
                    f"  Step {step:4d} | "
                    f"Coverage: {last_coverage.mean().item() * 100:5.1f}% | "
                    f"Captures: {last_captures.mean().item():.1f}"
                )

        # ── Episode summary ───────────────────────────────────────────────────
        print(f"\n  Episode {episode + 1} Results:")
        print(f"    Steps:            {step}")
        print(f"    Total reward:     {total_reward.mean().item():.2f}")
        print(f"    Final coverage:   {last_coverage.mean().item() * 100:.1f}%")
        print(f"    Captures used:    {last_captures.mean().item():.1f}")
        print(f"    Multi-view score: {last_multiview.mean().item():.2f}")
        print(f"    Voxels seen:      {last_voxels_seen} / {tracker.num_voxels}")

        # ── Save poses to JSON ────────────────────────────────────────────────
        if args_cli.save_poses and captured_poses:
            out_path = Path(args_cli.poses_dir) / f"episode_{episode + 1:02d}_poses.json"
            with open(out_path, "w") as f:
                json.dump({
                    "checkpoint":     args_cli.checkpoint,
                    "episode":        episode + 1,
                    "total_captures": len(captured_poses),
                    "final_coverage": float(last_coverage.mean().item()),
                    "poses":          captured_poses,
                }, f, indent=2)
            print(f"    Poses saved:      {out_path}  ({len(captured_poses)} captures)")

        elif args_cli.save_poses and not captured_poses:
            print(f"    Poses saved:      (no captures fired this episode)")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()