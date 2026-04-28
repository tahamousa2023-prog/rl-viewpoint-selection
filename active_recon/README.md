# UR5e Active Reconstruction

Reinforcement learning environment for active 3D reconstruction with UR5e robot arm.

## Task Description

The robot learns to capture **8 images** from **diverse viewpoints** around an object for optimal 3D reconstruction.

```
Episode Flow:
┌───────────────────────────────────────────────────────┐
│  Start → Move to viewpoint 1 → Capture                │
│       → Move to viewpoint 2 → Capture                 │
│       → ...                                           │
│       → Move to viewpoint 8 → Capture → Episode End   │
└───────────────────────────────────────────────────────┘
```

## Installation

```bash
# Navigate to project folder
cd /home/AP_PathMatters/UR5eActiveRecon

# Install the package
pip install -e source/active_recon

# Verify installation
python scripts/list_envs.py
```

## Usage

### Train

```bash
# With visualization (slow, for debugging)
python scripts/rsl_rl/train.py --task ActiveScan-UR5e-v0 --num_envs 64

# Headless (fast training)
python scripts/rsl_rl/train.py --task ActiveScan-UR5e-v0 --num_envs 1024 --headless
```

### Monitor Training

```bash
# In separate terminal
tensorboard --logdir logs/rsl_rl/ActiveScan-UR5e-v0
```

### Play Trained Policy

```bash
python scripts/rsl_rl/play.py \
    --task ActiveScan-UR5e-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/ActiveScan-UR5e-v0/<timestamp>/model_2000.pt
```

## Configuration

### Paths (edit in `env_cfg.py`)

```python
# Your robot USD
UR5E_USD_PATH = "/home/AP_PathMatters/path_matters/Isaacsim/envs/ur5e_w_cam_w_plane.usd"

# Your object USD  
OBJECT_USD_PATH = "/home/AP_PathMatters/path_matters/datasets/yoda/Baby_Yoda_v2.usd"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 1024 | Parallel environments |
| `episode_length_s` | 30.0 | Max seconds per episode |
| `required_captures` | 8 | Images to capture |
| `min_angular_distance` | 30.0° | Min angle between viewpoints |

## Reward Structure

| Reward | Weight | Description |
|--------|--------|-------------|
| `viewpoint_diversity` | +2.0 | Diverse viewing angles |
| `task_completion` | +10.0 | Bonus for 8 captures |
| `object_in_frame` | +1.0 | Object visible when capturing |
| `camera_facing_object` | +0.5 | Camera pointing at object |
| `optimal_distance` | -0.3 | Penalty for wrong distance |
| `action_rate` | -0.01 | Smooth motion |
| `collision_penalty` | -10.0 | Avoid collisions |

## Project Structure

```
UR5eActiveRecon/
├── source/
│   └── active_recon/
│       ├── pyproject.toml
│       └── active_recon/
│           ├── __init__.py
│           ├── config/
│           │   └── extension.toml
│           └── tasks/
│               └── active_scan/
│                   ├── __init__.py
│                   ├── env_cfg.py        # Main environment config
│                   ├── mdp/
│                   │   └── __init__.py   # Observations, rewards, etc.
│                   └── agents/
│                       └── rsl_rl_ppo_cfg.py
├── scripts/
│   ├── list_envs.py
│   └── rsl_rl/
│       ├── train.py
│       └── play.py
└── README.md
```

## Next Steps

1. **Test basic training** to verify setup works
2. **Adjust USD paths** if needed (joint names, camera position)
3. **Tune rewards** based on observed behavior
4. **Add reconstruction metric** to reward function
5. **Domain randomization** for multiple objects
