from typing import Dict, Any, Optional
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

class NonstationaryEnv(gym.Wrapper):
    """Creates controlled non-stationarity in Gymnasium environments"""
    
    def __init__(self, 
                env: gym.Env,
                shift_config: Dict[str, Any],
                seed: Optional[int] = None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.shift_config = shift_config
        self.original_params = self._get_modifiable_params()
        self.current_params = self._initialize_params()

    def _get_modifiable_params(self) -> Dict[str, Any]:
        """Environment-specific parameter extraction with validation"""
        try:
            if 'CartPole' in self.env.spec.id:
                return {
                    'gravity': self.env.gravity,
                    'masscart': self.env.masscart,
                    'masspole': self.env.masspole
                }
            elif 'HalfCheetah' in self.env.spec.id:
                return {
                    'body_mass': self.env.model.body_mass,
                    'dof_damping': self.env.model.dof_damping
                }
            raise NotImplementedError(f"Unsupported environment: {self.env.spec.id}")
        except AttributeError as e:
            raise ValueError("Environment doesn't support parameter modification") from e

    def _apply_parametric_shift(self):
        """Apply controlled parameter drift with bounds checking"""
        for param, config in self.shift_config.items():
            current = self.current_params[param]
            noise = self.rng.normal(config['mu'], config['sigma'])
            new_value = np.clip(
                current * (1 + noise),
                config['min'],
                config['max']
            )
            self.current_params[param] = new_value
        self._set_env_params(self.current_params)

    def reset(self, **kwargs):
        self._apply_parametric_shift()
        return super().reset(**kwargs)