# UR5e Active Reconstruction — RL Training

Early-stage RL environment for active scanning with camera integration.
Predecessor to UR5eVolumRecon — key lessons learned here informed the
final volumetric coverage approach.

## Setup

    conda activate env_isaaclab
    cd ~/UR5eActiveRecon

## Train

    python scripts/rsl_rl/train.py \
      --task ActiveScan-UR5e-v0 \
      --num_envs 32 \
      --max_iterations 2500

## Test with cameras enabled

    python scripts/check_setup.py --enable_cameras

## Development History

**29 Jan** — validate_camera_view, test_rewards, find_yoda scripts added.
Ran RL for 100 iterations. No camera used yet. Results promising but
object visibility not confirmed.

**30 Jan** — Updated reward function to reward image captures.
Camera still not saving images.

**31 Jan** — Training without camera is faster.
Note: comment out camera in env_cfg.py for training speed.

    python scripts/rsl_rl/train.py \
      --task ActiveScan-UR5e-v0 \
      --num_envs 32 \
      --max_iterations 500

**01 Feb** — Random image capture every 10 iterations added.
Camera now integrated into training loop.

## Key Difference vs UR5eVolumRecon

This repo uses image-based rewards. UR5eVolumRecon switched to
geometric frustum-based voxel coverage for speed — removing the
need to render images during training. That change enabled 16
parallel environments and stable PPO convergence.
