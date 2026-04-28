"""
Training script for Volume Reconstruction environment.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../agents"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Volume Reconstruction agent")
parser.add_argument("--task", type=str, default="VolumeRecon-UR5e-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=None, help="Max training iterations")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from datetime import datetime

# Try different import patterns for RSL-RL
try:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner, RslRlVecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner
        from isaaclab_rl.rsl_rl.vecenv import RslRlVecEnvWrapper
    except ImportError:
        from rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import volume_recon.tasks
from volume_recon.tasks.volume_scan.env_cfg import VolumeReconEnvCfg
from rsl_rl_ppo_cfg import VolumeReconPPORunnerCfg


def main():
    env_cfg = VolumeReconEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    agent_cfg = VolumeReconPPORunnerCfg()
    
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    
    log_dir = os.path.join(
        os.path.dirname(__file__),
        "../../logs/rsl_rl",
        args_cli.task,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("VOLUME RECONSTRUCTION TRAINING")
    print("="*60)
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Max iterations: {agent_cfg.max_iterations}")
    print(f"Log dir: {log_dir}")
    print("="*60 + "\n")
    
    runner = RslRlOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    
    if args_cli.checkpoint:
        print(f"Loading checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)
    
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    runner.save(os.path.join(log_dir, "model_final.pt"))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {log_dir}")
    print("="*60 + "\n")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()