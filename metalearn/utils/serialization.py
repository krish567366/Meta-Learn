import torch
import hashlib
from pathlib import Path
from typing import Dict, Any

class ModelIO:
    """Safe model serialization with version control"""
    
    def save(self, 
            model: torch.nn.Module, 
            path: str,
            metadata: Dict[str, Any] = None,
            compression: str = 'zip') -> str:
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'state_dict': model.state_dict(),
            'metadata': metadata or {},
            'hash': self._compute_hash(model)
        }
        
        # Save with proper compression
        if compression == 'zip':
            torch.save(state, path.with_suffix('.zip'))
        else:
            torch.save(state, path)
            
        return str(path)

    def load(self, path: str, device: str = 'cpu') -> Dict[str, Any]:
        state = torch.load(path, map_location=device)
        
        if not self._verify_hash(state['state_dict'], state['hash']):
            raise ValueError("Model checksum verification failed")
            
        return state

    def _compute_hash(self, model: torch.nn.Module) -> str:
        hasher = hashlib.sha256()
        for p in model.parameters():
            hasher.update(p.cpu().detach().numpy().tobytes())
        return hasher.hexdigest()

    def _verify_hash(self, state_dict: Dict[str, Any], hash: str) -> bool:
        current_hash = self._compute_hash_from_state(state_dict)
        return current_hash == hash

    def _compute_hash_from_state(self, state_dict: Dict[str, Any]) -> str:
        hasher = hashlib.sha256()
        for p in state_dict.values():
            hasher.update(p.cpu().numpy().tobytes())
        return hasher.hexdigest()