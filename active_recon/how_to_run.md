conda activate env_isaaclab

cd UR5eActiveRecon

python scripts/rsl_rl/train.py --task ActiveScan-UR5e-v0 --num_envs 32 --max_iterations 25
