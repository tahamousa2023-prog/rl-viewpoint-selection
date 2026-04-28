"""
Play trained policy with visual feedback.
Run: python scripts/rsl_rl/play.py --checkpoint logs/rsl_rl/ActiveScan-UR5e-v0/model_500.pt
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="ActiveScan-UR5e-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--random", action="store_true", help="Use random policy")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

import active_recon.tasks
from active_recon.tasks.active_scan.env_cfg import ActiveScanEnvCfg


class SimpleActor(nn.Module):
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


def main():
    env_cfg = ActiveScanEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    
    # Load or skip model
    actor = None
    if not args_cli.random:
        print(f"[INFO] Loading: {args_cli.checkpoint}")
        checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        actor = SimpleActor(51, 7).to("cuda:0")
        actor_state = {k.replace("actor.", ""): v for k, v in state_dict.items() if "actor" in k}
        actor.actor.load_state_dict(actor_state)
        actor.eval()
        print("[INFO] Model loaded!")
    else:
        print("[INFO] Using RANDOM policy")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"].to("cuda:0").float()
    
    print("\n" + "="*70)
    print("PLAYING POLICY - Watch Isaac Sim window!")
    print("="*70)
    print("Legend:")
    print("  📸 = Capture triggered")
    print("  🎯 = Good alignment with object")
    print("  ⚠️  = Poor alignment")
    print("="*70 + "\n")
    
    episode = 0
    step = 0
    episode_reward = 0
    
    while simulation_app.is_running():
        try:
            with torch.no_grad():
                if args_cli.random:
                    actions = torch.randn(1, 7, device="cuda:0") * 0.3
                    if step % 25 == 20:  # Periodic captures
                        actions[0, 6] = 1.0
                else:
                    actions = actor(obs)
            
            obs_dict, reward, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"].to("cuda:0").float()
            
            episode_reward += reward[0].item()
            step += 1
            
            # Get state info
            scan_state = unwrapped._scan_state
            captures = scan_state["capture_count"][0].item()
            just_captured = scan_state["just_captured"][0].item()
            
            # Get robot info
            robot = unwrapped.scene["robot"]
            ee_idx = robot.find_bodies("wrist_3_link")[0][0]
            ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
            ee_quat = robot.data.body_quat_w[0, ee_idx].cpu().numpy()
            
            # Calculate alignment
            obj_pos = np.array([0.5, 0.0, 0.1])
            to_obj = obj_pos - ee_pos
            to_obj_norm = to_obj / (np.linalg.norm(to_obj) + 1e-8)
            
            # Camera forward from quaternion
            w, x, y, z = ee_quat
            cam_fwd = np.array([
                2 * (x*z + w*y),
                2 * (y*z - w*x),
                1 - 2*(x*x + y*y)
            ])
            alignment = np.dot(cam_fwd, to_obj_norm)
            distance = np.linalg.norm(to_obj)
            
            # Print status
            if just_captured:
                align_icon = "🎯" if alignment > 0.7 else "⚠️"
                print(f"📸 Capture {captures}/8 | Step {step:4d} | "
                      f"Align: {alignment:.2f} {align_icon} | "
                      f"Dist: {distance:.2f}m | "
                      f"Pos: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]")
            elif step % 50 == 0:
                cap_action = actions[0, 6].item()
                print(f"   Step {step:4d} | Captures: {captures}/8 | "
                      f"Align: {alignment:.2f} | Dist: {distance:.2f}m | "
                      f"CaptureAction: {cap_action:+.2f}")
            
            # Episode end
            done = terminated.any().item() or truncated.any().item()
            if done:
                episode += 1
                print(f"\n{'─'*70}")
                print(f"Episode {episode} Complete!")
                print(f"  Total Reward: {episode_reward:.2f}")
                print(f"  Captures: {captures}/8")
                print(f"  Steps: {step}")
                print(f"{'─'*70}\n")
                
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"].to("cuda:0").float()
                step = 0
                episode_reward = 0
                
        except KeyboardInterrupt:
            break
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()