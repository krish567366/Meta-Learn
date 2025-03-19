from typing import Dict, Tuple
import torch
from torch import nn, Tensor

class HyperNetwork(nn.Module):
    """Efficient hypernetwork with weight-sharing"""
    def __init__(self, 
                input_dim: int,
                output_shapes: Dict[str, Tuple[int]]):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(512, int(torch.prod(torch.tensor(shape))))
            for name, shape in output_shapes.items()
        })
        self.output_shapes = output_shapes

    def forward(self, context: Tensor) -> Dict[str, Tensor]:
        shared = self.shared_mlp(context)
        return {
            name: head(shared).view(-1, *shape)
            for name, (head, shape) in zip(
                self.output_shapes.keys(),
                self.heads.items(),
                self.output_shapes.values()
            )
        }

class NeuralOptimizer(nn.Module):
    """Learned optimization rule with gating"""
    def __init__(self, param_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(param_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.delta_net = nn.Sequential(
            nn.Linear(param_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)
        )

    def forward(self, grad: Tensor, param: Tensor) -> Tensor:
        inp = torch.cat([grad.flatten(), param.flatten()])
        gate = self.gate(inp)
        delta = self.delta_net(inp)
        return gate * delta
    
class HyperDimensionalAdapter(nn.Module):
    """4D hypernetwork with temporal-spatial disentanglement"""
    
    def __init__(self, 
                input_dim: int, 
                output_shapes: Dict[str, Tuple[int]]):
        super().__init__()
        self.hypercube = nn.Parameter(torch.randn(16, 16, 16, 16))
        self.projection_heads = nn.ModuleDict({
            name: nn.Sequential(
                HyperFourierLayer(4, 3),
                AdaptiveWaveletTransform(output_dim=np.prod(shape))
            ) for name, shape in output_shapes.items()
        })

    def forward(self, context: Tensor) -> Dict[str, Tensor]:
        # 4D tensor slicing based on context
        slice_indices = (context * 15).long()
        hyper_slices = self.hypercube[
            slice_indices[..., 0],
            slice_indices[..., 1],
            slice_indices[..., 2],
            :
        ]
        return {
            name: head(hyper_slices).view(shape)
            for name, (head, shape) in zip(
                self.output_shapes.keys(),
                self.projection_heads.items(),
                self.output_shapes.values()
            )
        }

class HyperFourierLayer(nn.Module):
    """Learnable frequency domain transformations"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.freq_weights = nn.Parameter(torch.randn(output_dim, input_dim))
        self.phase_shift = nn.Parameter(torch.randn(output_dim))
        
    def forward(self, x: Tensor) -> Tensor:
        x_fft = torch.fft.rfft(x, dim=-1)
        modulated = x_fft * torch.complex(self.freq_weights, self.phase_shift)
        return torch.fft.irfft(modulated, dim=-1)