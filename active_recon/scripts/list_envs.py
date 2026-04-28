#!/usr/bin/env python3
"""List all registered environments."""

import argparse
import os
import sys

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source", "active_recon"))

from isaaclab.app import AppLauncher

# Create minimal app
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import gym and our module
import gymnasium as gym
import active_recon  # noqa: F401


def main():
    print("\n" + "=" * 60)
    print("Active Recon - Registered Environments")
    print("=" * 60 + "\n")
    
    all_envs = gym.envs.registry.keys()
    our_envs = [env for env in all_envs if "ActiveScan" in env]
    
    if our_envs:
        for env_id in sorted(our_envs):
            print(f"  • {env_id}")
    else:
        print("  No environments found!")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
