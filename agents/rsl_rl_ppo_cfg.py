"""
RSL-RL PPO agent configuration for Volume Reconstruction.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class VolumeReconPPORunnerCfg:
    """Configuration for the PPO runner."""
    
    # Runner settings
    num_steps_per_env: int = 24
    max_iterations: int = 2000
    save_interval: int = 100
    experiment_name: str = "volume_recon_ur5e"
    empirical_normalization: bool = False
    
    # Policy network
    policy_class_name: str = "ActorCritic"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 1.0
    
    # Algorithm
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for RSL-RL runner."""
        return {
            # Runner config
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "save_interval": self.save_interval,
            "experiment_name": self.experiment_name,
            "empirical_normalization": self.empirical_normalization,
            
            # REQUIRED: observation groups config
            "obs_groups": {
                "policy": ["policy"],
            },
            
            # Policy config
            "policy": {
                "class_name": self.policy_class_name,
                "init_noise_std": self.init_noise_std,
                "actor_hidden_dims": self.actor_hidden_dims,
                "critic_hidden_dims": self.critic_hidden_dims,
                "activation": self.activation,
            },
            
            # Algorithm config
            "algorithm": {
                "class_name": "PPO",
                "value_loss_coef": self.value_loss_coef,
                "use_clipped_value_loss": self.use_clipped_value_loss,
                "clip_param": self.clip_param,
                "entropy_coef": self.entropy_coef,
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "learning_rate": self.learning_rate,
                "schedule": self.schedule,
                "gamma": self.gamma,
                "lam": self.lam,
                "desired_kl": self.desired_kl,
                "max_grad_norm": self.max_grad_norm,
            },
        }