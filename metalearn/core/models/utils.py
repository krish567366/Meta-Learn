import torch

def create_mask(parameter: torch.Tensor, 
            sparsity: float) -> torch.Tensor:
    """Create binary mask for parameter pruning"""
    if sparsity <= 0:
        return torch.ones_like(parameter)
    
    flat = parameter.flatten()
    k = int((1 - sparsity) * flat.numel())
    threshold = torch.topk(flat.abs(), k)[0][-1]
    return (parameter.abs() >= threshold).float()

def elastic_weight_update(original: torch.Tensor,
                        new: torch.Tensor,
                        elasticity: float) -> torch.Tensor:
    """Blend parameters with elastic consolidation"""
    return elasticity * new + (1 - elasticity) * original

def freeze_parameters(model: torch.nn.Module,
                    patterns: List[str]):
    """Freeze parameters matching name patterns"""
    for name, param in model.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad_(False)