from swarm import Swarm
from typing import Tuple, Optional
from custom_typings import ObjectiveFunction, Vector, Fitness
import numpy as np


class MultiSwarmOptimizer:
    def __init__(
        self,
        objective: ObjectiveFunction,
        dim: int,
        bounds: Tuple[float, float],
        num_swarms: int = 3,
        swarm_size: int = 20,
        max_generations: int = 100,
        stop_score: float = 1e-6,
        migration_rate: float = 0.1
    ) -> None:
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.max_generations = max_generations
        self.stop_score = stop_score
        self.migration_rate = migration_rate
        self.swarms = [Swarm(objective, dim, bounds, swarm_size) for _ in range(num_swarms)]

    def migrate_solutions(self) -> None:
        """Migrate top solutions between swarms in a vectorized way."""
        migrants_per_swarm = max(1, int(self.swarm_size * self.migration_rate))

        migrant_positions = []
        source_indices = []

        # Recolectar migrantes y de qué swarm vienen
        for swarm_idx, swarm in enumerate(self.swarms):
            top = swarm.memory.get_top_k(migrants_per_swarm)
            for sol in top:
                migrant_positions.append(sol)
                source_indices.append(swarm_idx)

        if not migrant_positions:
            return

        migrant_positions = np.array(migrant_positions)
        migrant_values = self.objective(migrant_positions)  # vectorizado

        # Distribuir migrantes (evitando devolverlos a su swarm de origen)
        for i, swarm in enumerate(self.swarms):
            for _, (pos, val, src_idx) in enumerate(zip(migrant_positions, migrant_values, source_indices)):
                if i != src_idx:
                    swarm.memory.update(pos, val)

    def optimize(self) -> Tuple[Vector, Fitness]:
        best_global: Optional[Vector] = None
        best_score: float = float('inf')

        for gen in range(self.max_generations):
            self.migrate_solutions()

            # Global best tracking
            for swarm in self.swarms:
                swarm.step()
                if swarm.best_score < best_score:
                    best_score = swarm.best_score
                    best_global = swarm.best_global[:]

            print(f"Generation {gen+1} -> Mejor global hasta ahora: {best_score:.6f}")

            if best_score <= self.stop_score:
                print("Optimal Solution found... Stopping optimization process...")
                return best_global, best_score

        return best_global, best_score