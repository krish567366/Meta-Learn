from typing import Dict, List, Any
import torch

class FederatedMetaLearner:
    """Federated version of EMCC algorithm"""
    def __init__(self,
                global_model: torch.nn.Module,
                clients: List[Any],
                aggregation: str = 'fedavg'):
        self.global_model = global_model
        self.clients = clients
        self.aggregation = aggregation

    def aggregate_updates(self, 
                        client_updates: List[Dict[str, torch.Tensor]]):
        """Aggregate client meta-updates"""
        if self.aggregation == 'fedavg':
            return {
                k: torch.mean(torch.stack([u[k] for u in client_updates]), dim=0)
                for k in client_updates[0].keys()
            }
        elif self.aggregation == 'fedprox':
            # Implement FedProx aggregation
            pass

    def run_round(self, 
                num_clients: int,
                local_steps: int) -> Dict[str, Any]:
        sampled_clients = np.random.choice(
            self.clients, num_clients, replace=False)
        
        client_updates = []
        for client in sampled_clients:
            local_update = client.local_train(
                self.global_model.state_dict(),
                local_steps
            )
            client_updates.append(local_update)
            
        global_update = self.aggregate_updates(client_updates)
        self.global_model.load_state_dict(global_update)
        
        return {'num_clients': num_clients}