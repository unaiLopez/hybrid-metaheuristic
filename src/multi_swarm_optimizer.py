from swarm import Swarm
from typing import Tuple, Optional, Callable, List
import numpy as np
import local_search


class MultiSwarmOptimizer:
    def __init__(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        dim: int,
        bounds: List[Tuple[float, float]],
        num_swarms: int = 3,
        swarm_size: int = 20,
        max_generations: int = 100,
        num_generations_no_improve_stop: int = 20,
        stop_score: float = 1e-6,
        migration_rate: float = 0.1
    ) -> None:
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.max_generations = max_generations
        self.num_generations_no_improve_stop = num_generations_no_improve_stop
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

    def optimize(self, no_improvement_threshold: float = 1e-5) -> Tuple[np.ndarray, float]:
        best_global: Optional[np.ndarray] = None
        best_score: float = float('inf')

        num_no_improve_generations = 0
        previous_score = None
        for gen in range(self.max_generations):
            self.migrate_solutions()

            for swarm in self.swarms:
                swarm.step()
                if swarm.best_score < best_score:
                    best_score = swarm.best_score
                    best_global = swarm.best_global[:]

            print(f"Generation {gen+1} -> Mejor global hasta ahora: {best_score:.10f}")

            if previous_score is not None:
                improvement = previous_score - best_score
                if improvement < no_improvement_threshold:
                    num_no_improve_generations += 1
                else:
                    num_no_improve_generations = 0
            previous_score = best_score

            if best_score <= self.stop_score:
                print("Optimal Solution found... Stopping optimization process...")
                return best_global, best_score
            
            if gen % 200 == 0 and gen != 0:
                # I have to work on this part local explotation with Adam Optimizer
                # 🔽 Adam refinement step
                for swarm in self.swarms:
                    print("Starting local optimization with Adam...")
                    best_global, best_score = local_search.adam_optimize(
                        f=self.objective,
                        x_init=swarm.best_global,
                        learning_rate=1e-1,
                        max_iters=500,
                        tol=1e-6
                    )
                    if best_score < swarm.best_score:
                        swarm.update_best(best_global, best_score)
                        print(f"Adam optimization completed. Final score: {best_score:.10f}")
                    if best_score <= self.stop_score:
                        print("Optimal Solution found... Stopping optimization process...")
                        return best_global, best_score

            if num_no_improve_generations >= self.num_generations_no_improve_stop:
                print(f"Improvement below {no_improvement_threshold} for {self.num_generations_no_improve_stop} generations. "
                    "Assuming convergence... Stopping optimization process.")
                return best_global, best_score


           
                    
        return best_global, best_score
