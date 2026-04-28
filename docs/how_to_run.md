# How to Run — RL Viewpoint Selection

## Setup

    cd ~/UR5eVolumRecon
    conda activate env_isaaclab

## Test environment (1 env, with visualization)

    python scripts/test_env.py --num_envs 1

## Train

    python scripts/rsl_rl/train.py \
      --task VolumeRecon-UR5e-v0 \
      --num_envs 32 \
      --max_iterations 2000 \
      --headless

## Continue from checkpoint

    python experiments/taha/run_experiment.py \
      --num_envs 16 \
      --max_iterations 5000 \
      --experiment_name exp_10 \
      --checkpoint logs/taha/exp_07_stricter_wasteful/2026-03-22_15-56-14/model_final.pt \
      --headless

## Monitor with TensorBoard

    tensorboard --logdir ~/UR5eVolumRecon/logs/rsl_rl/ --port 6006

Open: http://localhost:6006

## Evaluate policy + save poses

    python scripts/rsl_rl/play.py \
      --task VolumeRecon-UR5e-v0 \
      --num_envs 1 \
      --num_episodes 3 \
      --checkpoint logs/taha/exp_07_stricter_wasteful/2026-03-22_15-56-14/model_final.pt \
      --save-poses \
      --poses-dir outputs/poses/exp_07

## Key experiments

| Exp | Task success | Notes |
|-----|-------------|-------|
| exp_06 | 0.4% | Baseline, no shaping |
| exp_07 | 45.2% | + Proximity shaping |
| exp_09 | TBD | Stricter wasteful penalty |
| exp_10 | TBD | Continue from exp_07 |

## MDP

- State: 86D (joints, camera pose, coverage %, 8x8 voxel map)
- Action: 7D (6 joint deltas + capture trigger)
- Episode ends: 75% coverage OR 50 captures OR 1000 steps
- Algorithm: PPO, MLP [256,128,64], 16 parallel envs, gamma=0.99
