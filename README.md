# UR5e Volumetric Room Reconstruction

RL-based trajectory planning for 3D room reconstruction using a UR5e robot arm with camera.

## Goal

Train a robot to find optimal camera trajectories that maximize volumetric coverage of a room/workspace. The robot learns to capture images from viewpoints that:

1. **Maximize volume coverage** - see as much of the 3D space as possible
2. **Multi-view consistency** - view regions from multiple angles for better reconstruction
3. **Efficient trajectories** - minimize captures while maximizing coverage
4. **Smooth motion** - avoid jerky movements

## Key Difference from Object Scanning

- **No object dependency**: Coverage is based on the entire volume, not specific objects
- **Voxel-based tracking**: The room is divided into voxels, and we track which are visible
- **View angle diversity**: Each voxel benefits from being seen from multiple angles

## Installation

```bash
cd /home/AP_PathMatters/UR5eVolumeRecon
pip install -e source/volume_recon
```

## Training

```bash
python scripts/rsl_rl/train.py --task VolumeRecon-UR5e-v0 --num_envs 32 --max_iterations 1000
```

## Testing

```bash
python scripts/test_env.py --num_envs 1
```

## Configuration

Edit `source/volume_recon/volume_recon/tasks/volume_scan/env_cfg.py`:

- `VOLUME_BOUNDS`: Define the room dimensions to reconstruct
- `VOXEL_RESOLUTION`: Grid resolution (higher = more precise but slower)
- `REQUIRED_CAPTURES`: Target number of captures
- `COVERAGE_THRESHOLD`: Target coverage percentage to complete episode

## Architecture

```
Observation Space (dim varies by voxel resolution):
├── joint_pos (6)      - Robot joint positions
├── joint_vel (6)      - Robot joint velocities  
├── camera_pos (3)     - Camera world position
├── camera_quat (4)    - Camera orientation
├── coverage_pct (1)   - Current volume coverage percentage
├── capture_count (1)  - Number of captures taken
└── coverage_map (N)   - Downsampled voxel coverage grid

Action Space (7):
├── arm_action (6)     - Joint position targets
└── capture (1)        - Trigger capture (>0.5 = capture)

Rewards:
├── coverage_increase  - New voxels seen this capture (+)
├── multi_view_bonus   - Voxels seen from new angles (+)
├── coverage_progress  - Overall coverage improvement (+)
├── action_smoothness  - Penalize jerky motion (-)
├── capture_efficiency - Bonus for high coverage per capture (+)
└── task_completion    - Bonus when target coverage reached (+)
```
