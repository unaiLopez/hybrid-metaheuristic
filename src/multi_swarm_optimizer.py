import numpy as np
from swarm import Swarm
from typing import Callable, List, Tuple, Optional
import local_search


def shrink_bounds(bounds: List[Tuple[float, float]], positions: np.ndarray, factor: float = 3.0) -> List[Tuple[float, float]]:
    """
    Shrinks the bounds around the mean of given positions by factor * std.

    :param bounds: Original bounds [(low, high), ...]
    :param positions: Array of shape (num_samples, dim)
    :param factor: Multiplier for standard deviation
    :return: New list of bounds
    """
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    mean = positions.mean(axis=0)
    std = positions.std(axis=0)

    new_lower = np.clip(mean - factor * std, lower, upper)
    new_upper = np.clip(mean + factor * std, lower, upper)
    print(f"Shrink bounds: new_lower={new_lower}, new_upper={new_upper}")
    return list(zip(new_lower.tolist(), new_upper.tolist()))


class MultiSwarmOptimizer:
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        num_swarms: int = 3,
        swarm_size: int = 20,
        max_generations: int = 100,
        no_improve_stop: int = 20,
        stop_score: float = 1e-6,
        migration_rate: float = 0.1,
        migration_interval: int = 20,
        adam_interval: int = 200,
        shrink_rounds: int = 3,
    ) -> None:
        self.objective = objective
        self.dim = dim
        self.initial_bounds = bounds.copy()
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.max_generations = max_generations
        self.no_improve_stop = no_improve_stop
        self.stop_score = stop_score
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.shrink_rounds = shrink_rounds
        self.adam_interval = adam_interval

    def _init_swarm(self, bounds: List[Tuple[float, float]]) -> Swarm:
        print(f"Initializing swarm with bounds: {bounds}")
        return Swarm(self.objective, self.dim, bounds, self.swarm_size)

    def migrate_solutions(self):
        migrants = int(self.swarm_size * self.migration_rate)
        if migrants == 0:
            return
        positions, sources = [], []
        for idx, swarm in enumerate(self.swarms):
            for pos in swarm.memory.get_top_k(migrants):
                positions.append(pos)
                sources.append(idx)
        if not positions:
            return
        positions = np.array(positions)
        values = [self.objective(p) for p in positions]
        for target_idx, swarm in enumerate(self.swarms):
            for pos, val, src in zip(positions, values, sources):
                if target_idx != src:
                    swarm.memory.update(pos, val)

    def _optimize_phase(self) -> Tuple[np.ndarray, float]:
        best_global, best_score = None, float('inf')
        no_improve = 0
        prev_score = None
        print("Starting PSO phase")

        for gen in range(self.max_generations):
            # Optional migration at Adam interval
            if gen != 0 and gen % self.migration_interval == 0:
                print(f"   Migration solutions between swarms")
                self.migrate_solutions()

            # PSO step
            for swarm_idx, swarm in enumerate(self.swarms):
                swarm.step()
                if swarm.best_score < best_score:
                    best_score = swarm.best_score
                    best_global = swarm.best_global.copy()
            print(f"  Generation {gen+1}/{self.max_generations}, best_score: {best_score}")

            # Adam refinement
            if self.adam_interval > 0 and (gen + 1) % self.adam_interval == 0:
                print(f"  Applying Adam refinement to the best solution of every swarm")
                for swarm_idx, swarm in enumerate(self.swarms):
                    x_refined, v_refined = local_search.adam_optimize(
                        f=self.objective,
                        x_init=swarm.best_global,
                        learning_rate=1e-1,
                        max_iters=500,
                        tol=1e-6
                    )
                    print(f"    Swarm {swarm_idx} Adam refined value: {v_refined}")
                    if v_refined < swarm.best_score:
                        swarm.update_best(x_refined, v_refined)
                    if v_refined < best_score:
                        best_score = v_refined
                        best_global = x_refined.copy()

            # Early stopping conditions
            if prev_score is not None and prev_score - best_score < 1e-5:
                no_improve += 1
            else:
                no_improve = 0
            prev_score = best_score

            if best_score <= self.stop_score or no_improve >= self.no_improve_stop:
                print("Ending PSO phase early")
                break

        return best_global, best_score

    def optimize(self) -> Tuple[np.ndarray, float]:
        # Initialize swarms with shared initial bounds
        self.swarms = [self._init_swarm(self.initial_bounds) for _ in range(self.num_swarms)]
        overall_best, overall_score = None, float('inf')

        for round_idx in range(self.shrink_rounds):
            print(f"=== Round {round_idx+1}/{self.shrink_rounds} ===")
            # Run PSO + periodic Adam
            best_x, best_val = self._optimize_phase()

            # Update overall best
            if best_val < overall_score:
                overall_best, overall_score = best_x, best_val

            # Early stopping if optimum reached
            if overall_score <= self.stop_score:
                print(f"Optimal solution reached (value={overall_score}). Exiting optimization.")
                return overall_best, overall_score

            # Prepare next-round swarms with individual shrunk bounds
            if round_idx < self.shrink_rounds - 1:
                new_swarms = []
                for swarm in self.swarms:
                    tops = np.vstack(swarm.memory.get_top_k(50))
                    sw_bounds = shrink_bounds(self.initial_bounds, tops)
                    new_swarms.append(self._init_swarm(sw_bounds))
                self.swarms = new_swarms

        print(f"Optimization complete. Best value: {overall_score}")
        return overall_best, overall_score