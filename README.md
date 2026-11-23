# RL Viewpoint Selection: Volumetric Coverage with a UR5e in Isaac Lab

Reinforcement learning for camera viewpoint selection, developed as my contribution to the [Path Matters](https://github.com/tahamousa2023-prog/path-matters-robotics) project at TU Berlin (WiSe 2025/26). A UR5e with a wrist-mounted camera learns, with PPO in NVIDIA Isaac Lab, to move to viewpoints that maximise volumetric coverage of the scanning workspace.

## Key result

| Experiment | Reward design | Task success (≥75% coverage) |
|---|---|---|
| exp_06 | Sparse coverage reward only | 0.4% |
| exp_07 | Coverage + dense proximity shaping | **45.2%** |

With coverage rewards alone, the robot rarely reached the scanning volume, so the reward almost never fired. Adding a continuous proximity gradient toward the workspace solved this sparse-reward problem: the agent first learns to approach the volume, then to scan it. The trained policy reached 75%+ coverage per episode, compared with 20.6% for random exploration. All results are from simulation.

## Environment

| | |
|---|---|
| Simulator | NVIDIA Isaac Sim + Isaac Lab |
| Algorithm | PPO (RSL-RL), 16 parallel environments |
| Policy | MLP actor-critic, hidden layers [256, 256, 128] |
| Observation | Joint positions and velocities, camera pose, coverage percentage, capture count, downsampled voxel coverage map |
| Action | 6 joint position targets + capture trigger |

**Reward terms:** new voxels seen, multi-view bonus for new viewing angles, coverage progress, capture efficiency, task completion bonus, action smoothness penalty, and (from exp_07) proximity shaping toward the scanning volume.

## Repository structure

```
agents/          PPO runner configuration (RSL-RL)
experiments/     reward functions, experiment runner, evaluation pipeline
scripts/         training, evaluation (play) and environment tests
active_recon/    earlier prototype: fixed 8-capture active scanning task
docs/            command reference for training, TensorBoard and evaluation
```

The Isaac Lab task package itself (environment and scene definition) is part of the team project setup and is not included here. This repository contains my training configuration, reward design, experiment scripts and evaluation tooling.

## Usage

```bash
conda activate env_isaaclab

# train an experiment
python experiments/run_experiment.py --num_envs 16 --max_iterations 5000 --experiment_name exp_10 --headless

# evaluate a trained policy and save the camera poses
python scripts/rsl_rl/play.py --task VolumeRecon-UR5e-v0 --num_envs 1 --num_episodes 3 \
  --checkpoint logs/<experiment>/<timestamp>/model_final.pt --save-poses
```

See [docs/how_to_run.md](docs/how_to_run.md) for the full command reference.

## Related work

The project led to a co-authored journal paper, currently under review at *The International Journal of Advanced Manufacturing Technology* (Springer): *Camera-Orientation Effects in Robotic Viewpoint Acquisition for Feed-Forward 3D Reconstruction in Manufacturing Inspection Workcells*.

## Contact

**Taha Mohammed** · [Website](https://tahamousa2023-prog.github.io) · [LinkedIn](https://www.linkedin.com/in/taha-mahmoud)
