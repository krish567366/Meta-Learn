from omegaconf import OmegaConf
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Hierarchical configuration system with environment overrides"""
    
    def __init__(self, base_path: str = 'configs/base.yaml'):
        self.base_config = self._load_config(base_path)
        self._apply_environment_overrides()
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            return OmegaConf.create(yaml.safe_load(f))
        
    def _apply_environment_overrides(self):
        env = os.getenv('METALEARN_ENV', 'development')
        env_config_path = f'configs/{env}.yaml'
        
        if Path(env_config_path).exists():
            env_config = self._load_config(env_config_path)
            self.config = OmegaConf.merge(self.base_config, env_config)
            
    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self.config, key, default=default)
    
    def update(self, updates: Dict[str, Any]):
        self.config = OmegaConf.merge(self.config, OmegaConf.create(updates))