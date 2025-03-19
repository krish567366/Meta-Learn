class QuantumDeviceManager:
    """Hybrid quantum-classical computing orchestration"""
    
    def __init__(self, qpu_backend: str = 'ionq_harmony'):
        self.qpu = QuantumProcessingUnit(backend=qpu_backend)
        self.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def hybrid_execution(self, 
                        quantum_circuit: QuantumCircuit,
                        classical_net: nn.Module) -> Tensor:
        # Split computation between QPU and GPU
        quantum_result = self.qpu.execute(quantum_circuit)
        classical_input = self._convert_quantum_output(quantum_result)
        return classical_net(classical_input.to(self.gpu))
    
    def _convert_quantum_output(self, result: QuantumState) -> Tensor:
        # Convert qubit measurements to tensor
        probabilities = result.get_probabilities()
        return torch.tensor([probabilities.get(f"{i:0{self.qpu.num_qubits}b}", 0)
                            for i in range(2**self.qpu.num_qubits)])