import logging
from typing import Dict, Any
import torch
import wandb
from tensorboardX import SummaryWriter

class MetaLogger:
    """Unified logging system with multiple backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self._init_console_logging(config)
        self.wandb_enabled = config.get('wandb', False)
        self.tb_writer = SummaryWriter(config['log_dir'])
        
        if self.wandb_enabled:
            wandb.init(project=config['project'], config=config)

    def _init_console_logging(self, config):
        logging.basicConfig(
            level=config.get('log_level', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('metalearn')

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all available backends"""
        # TensorBoard
        for name, value in metrics.items():
            self.tb_writer.add_scalar(name, value, step)
        
        # Weights & Biases
        if self.wandb_enabled:
            wandb.log(metrics, step=step)
        
        # Console
        self.logger.info(f"Step {step}: {metrics}")

    def log_model(self, model: torch.nn.Module, metrics: Dict[str, float]):
        """Save model checkpoints with metadata"""
        if self.wandb_enabled:
            artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
            torch.save(model.state_dict(), 'model.pth')
            artifact.add_file('model.pth')
            wandb.log_artifact(artifact, aliases=['latest'] + 
                            (['best'] if metrics.get('best', False) else []))