import random
import local_search

import numpy as np

from swarm import Swarm
from typing import Callable, List, Tuple


class MultiSwarmOptimizer:
    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        bounds: List[Tuple[float, float]],
        num_swarms: int = 3,
        swarm_size: int = 200,
        top_social_bests_to_choose_per_swarm: int = 10,
        c1_range_per_swarm: Tuple[float, float] = (1.0, 1.5),
        c2_range_per_swarm: Tuple[float, float] = (1.0, 1.5),
        inertia_range_per_swarm: Tuple[float, float] = (0.4, 0.8),
        max_generations: int = 100,
        no_improve_stop: int | None = 20,
        stop_score: float | None = 1e-6,
        migration_rate: float = 0.05,
        migration_interval: int = 20,
        adam_interval: int = 200,
        shrink_rounds: int = 3,
        top_k_for_next_round_bounds: int = 50,
        min_relative_variance_before_reseeding: float = 0.01
    ) -> None:
        self.objective = objective
        self.dim = dim
        self.initial_bounds = bounds.copy()
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.top_social_bests_to_choose_per_swarm = top_social_bests_to_choose_per_swarm
        self.c1_range_per_swarm = c1_range_per_swarm
        self.c2_range_per_swarm = c2_range_per_swarm
        self.inertia_range_per_swarm = inertia_range_per_swarm
        self.max_generations = max_generations
        self.no_improve_stop = no_improve_stop
        self.stop_score = stop_score
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.adam_interval = adam_interval
        self.shrink_rounds = shrink_rounds
        self.top_k_for_next_round_bounds = top_k_for_next_round_bounds
        self.min_relative_variance_before_reseeding=min_relative_variance_before_reseeding

    def _init_swarm(self, bounds: List[Tuple[float, float]]) -> Swarm:
        print(f"Initializing swarm with bounds: {bounds}")
        return Swarm(
            objective=self.objective,
            dim=self.dim,
            bounds=bounds,
            swarm_size=self.swarm_size,
            top_social_bests_to_choose=self.top_social_bests_to_choose_per_swarm,
            c1_range=self.c1_range_per_swarm,
            c2_range=self.c2_range_per_swarm,
            inertia_range=self.inertia_range_per_swarm,
            min_relative_variance_before_reseeding=self.min_relative_variance_before_reseeding
        )
    
    def _shrink_bounds(self, bounds: List[Tuple[float, float]], positions: np.ndarray, best_global: np.ndarray, factor: float = 3.0, min_shrink_ratio: float = 0.05) -> List[Tuple[float, float]]:
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        mean = positions.mean(axis=0)
        std = positions.std(axis=0)

        min_width = min_shrink_ratio * (upper - lower)
        half_width = np.maximum(factor * std, min_width / 2)

        new_lower = np.clip(mean - half_width, lower, upper)
        new_upper = np.clip(mean + half_width, lower, upper)

        new_lower = np.minimum(new_lower, best_global)
        new_upper = np.maximum(new_upper, best_global)

        return list(zip(new_lower.tolist(), new_upper.tolist()))

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
            # Migrate solutions between swarms
            if gen != 0 and gen % self.migration_interval == 0:
                print(f"   Migration solutions between swarms")
                self.migrate_solutions()

            # PSO step
            for swarm_idx, swarm in enumerate(self.swarms):
                swarm.step(gen)
                if swarm.best_score < best_score:
                    best_score = swarm.best_score
                    best_global = swarm.best_global.copy()
                
            print(f"  Generation {gen+1}/{self.max_generations}, best_score: {best_score}")

            # Adam refinement
            if self.adam_interval > 0 and (gen + 1) % self.adam_interval == 0:
                print(f"  Applying Adam refinement to random a particle of every swarm")
                for swarm_idx, swarm in enumerate(self.swarms):
                    random_particle = random.choice(swarm.positions)
                    x_refined, v_refined = local_search.adam_optimize(
                        objective=self.objective,
                        x_init=random_particle,
                        learning_rate=1e-1,
                        max_iters=500,
                        tol=1e-6
                    )
                    print(f"    Swarm {swarm_idx} Adam refined value: {v_refined}")

                    worst_index = np.argmax(swarm.values)
                    worst_score = swarm.values[worst_index]

                    if v_refined < worst_score:
                        swarm.positions[worst_index] = x_refined
                        swarm.values[worst_index] = v_refined

                        if v_refined < swarm.best_score:
                            swarm.update_best(x_refined, v_refined)

                    if v_refined < best_score:
                        best_score = v_refined
                        best_global = x_refined.copy()

            # Early stopping conditions
            if self.no_improve_stop is not None:
                if prev_score is not None and prev_score - best_score < 1e-5:
                    no_improve += 1
                else:
                    no_improve = 0
                prev_score = best_score

                if no_improve >= self.no_improve_stop:
                    print(f"Ending PSO phase early. No improvement for {self.no_improve_stop} generations.")
                    break

            if self.stop_score is not None:
                if best_score <= self.stop_score:
                    print(f"Ending PSO phase early. Stop score {self.stop_score} reached.")
                    break

        return best_global, best_score

    def optimize(self) -> Tuple[np.ndarray, float]:
        self.swarms = [self._init_swarm(self.initial_bounds) for _ in range(self.num_swarms)]
        overall_best, overall_score = None, float('inf')

        for round_idx in range(self.shrink_rounds):
            print(f"=== Round {round_idx+1}/{self.shrink_rounds} ===")
            # Run PSO + periodic Adam
            best_x, best_val = self._optimize_phase()

            if best_val < overall_score:
                overall_best, overall_score = best_x, best_val

            # Early stopping if optimum reached
            if self.stop_score is not None:
                if overall_score <= self.stop_score:
                    print(f"Optimal solution reached (value={overall_score}). Exiting optimization.")
                    return overall_best, overall_score

            # Prepare next-round swarms with individual shrunk bounds
            if round_idx < self.shrink_rounds - 1:
                new_swarms = []
                for swarm in self.swarms:
                    tops = np.vstack(swarm.memory.get_top_k(self.top_k_for_next_round_bounds))
                    sw_bounds = self._shrink_bounds(self.initial_bounds, tops, swarm.best_global)
                    new_swarms.append(self._init_swarm(sw_bounds))
                self.swarms = new_swarms

        print(f"Optimization complete. Best value: {overall_score}")
        return overall_best, overall_score