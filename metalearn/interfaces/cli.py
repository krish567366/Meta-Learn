import click

@click.group()
def cli():
    """Metalearn Command Line Interface"""
    pass

@cli.command()
@click.option('--config', '-c', default='configs/base.yaml')
def train(config: str):
    """Launch meta-training pipeline"""
    click.echo(f"Starting training with config: {config}")
    # Implementation would load config and start training

@cli.command()
@click.argument('model_path')
def serve(model_path: str):
    """Deploy model as REST API"""
    click.echo(f"Launching API service for model: {model_path}")