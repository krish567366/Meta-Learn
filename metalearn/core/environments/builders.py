class DarwinianTaskBuilder:
    """Task generator with evolutionary pressure dynamics"""
    
    def __init__(self, 
                base_tasks: List[Dict],
                mutation_rate: float = 0.01,
                selection_pressure: float = 0.7):
        self.population = base_tasks
        self.mutation_rate = mutation_rate
        self.selection = TournamentSelection(pressure=selection_pressure)
        
    def evolve_generation(self, fitness_scores: List[float]):
        # Evolutionary algorithm steps
        selected = self.selection.select(self.population, fitness_scores)
        crossed_over = self.crossover(selected)
        mutated = self.mutate(crossed_over)
        self.population = mutated
        
    def crossover(self, parents: List) -> List:
        return [self._single_crossover(p1, p2) 
                for p1, p2 in zip(parents[::2], parents[1::2])]
    
    def _single_crossover(self, p1: Dict, p2: Dict) -> Dict:
        return {k: p1[k] if torch.rand(1) < 0.5 else p2[k] 
                for k in p1.keys()}
    
    def mutate(self, population: List) -> List:
        return [{
            k: v + torch.randn_like(v) * self.mutation_rate
            if torch.rand(1) < 0.2 else v 
            for k, v in task.items()
        } for task in population]