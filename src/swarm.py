import random
import numpy as np
import local_search as local_search

from typing import Tuple
from memory_bank import MemoryBank
from typing import Callable, List

class Swarm:
    def __init__(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        dim: int,
        bounds: List[Tuple[float, float]],
        swarm_size: int,
        top_social_bests_to_choose: int = 10,
        c1_range: Tuple[float, float] = (1.0, 1.5),
        c2_range: Tuple[float, float] = (1.0, 1.5),
        inertia_range: Tuple[float, float] = (0.4, 0.8),
        min_relative_variance_before_reseeding: float = 0.01
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])
        self.ranges = self.upper_bounds - self.lower_bounds
        self.top_social_bests_to_choose = top_social_bests_to_choose
        self.c1_range = c1_range
        self.c2_range = c2_range
        self.inertia_range = inertia_range
        self.min_relative_variance_before_reseeding = min_relative_variance_before_reseeding

        self.memory = MemoryBank()
        self.positions = np.array([np.random.uniform(low, high, size=swarm_size) for (low, high) in bounds]).T
        self.c1s = np.random.uniform(c1_range[0], c1_range[1], size=(swarm_size, 1))
        self.c2s = np.random.uniform(c2_range[0], c2_range[1], size=(swarm_size, 1))
        self.inertias = np.random.uniform(inertia_range[0], inertia_range[1], size=(swarm_size, 1))

        self.velocities = np.random.uniform(
            -0.1 * self.ranges,
            0.1 * self.ranges,
            size=(swarm_size, dim)
        )
        self.values = self.objective(self.positions)
        self.pbest_pos = self.positions.copy()
        self.pbest_val = self.values.copy()

        self.best_index = np.argmin(self.values)
        self.best_global = self.positions[self.best_index].copy()
        self.best_score = self.values[self.best_index]
        self.memory.update(self.best_global, self.best_score)
        self.diversity = self.relative_variance()
    
    def update_best(self, new_best_global: np.ndarray, new_best_value: float) -> None:
        self.values[self.best_index] = new_best_value
        self.positions[self.best_index] = new_best_global
        self.best_global = new_best_global
        self.best_score = new_best_value
        self.memory.update(self.best_global, self.best_score)

    def relative_variance(self) -> float:
        """
        Calcula la varianza relativa promedio entre la varianza actual
        del enjambre y la máxima varianza posible uniforme en cada dimensión.
        Retorna un valor entre 0 (sin diversidad) y 1 (máxima diversidad).
        """
        var_pos = self.positions.var(axis=0)
        var_uniform = ((self.upper_bounds - self.lower_bounds) ** 2) / 12
        relative_var = var_pos / var_uniform

        # En caso de que var_uniform sea 0 (limite con 0 ancho), evitar división por cero
        relative_var = np.where(var_uniform > 0, relative_var, 1.0)

        return np.mean(relative_var)


    def check_and_reseed_worst_particles(self) -> None:
        self.diversity = self.relative_variance()
        if self.diversity < self.min_relative_variance_before_reseeding:
            print(f"Relative variance is low ({self.diversity:.4f}), reseeding worst particles to augment the diversity...")
            num_particles = self.positions.shape[0]
            num_to_reseed = num_particles // 2

            # Obtain the ordered indices of the world particles
            worst_indices = np.argsort(self.values)[-num_to_reseed:]

            new_positions = np.array([np.random.uniform(low, high, size=num_to_reseed) for (low, high) in self.bounds]).T
            self.positions[worst_indices] = new_positions

            # Reinitialize velocities for those particles
            self.velocities[worst_indices] = np.random.uniform(
                -0.1 * self.ranges,
                0.1 * self.ranges,
                size=(num_to_reseed, self.dim)
            )

            # Evaluate again all the solutions / new positions
            # Update best personal solutions
            self.values[worst_indices] = self.objective(self.positions[worst_indices])
            mask = self.values[worst_indices] < self.pbest_val[worst_indices]
            self.pbest_val[worst_indices][mask] = self.values[worst_indices][mask]
            self.pbest_pos[worst_indices][mask] = self.positions[worst_indices][mask]


    def step(self, gen: int) -> None:
        # Update best global solution
        self.best_index = np.argmin(self.values)
        self.best_global = self.positions[self.best_index].copy()
        self.best_score = self.values[self.best_index]
        self.memory.update(self.best_global, self.best_score)

        # Get randomly one of the top k best social solutions (force cooperative exploration)
        social_best = np.array((
            random.choice(self.memory.get_top_k(self.top_social_bests_to_choose))
            if self.memory.solutions else self.best_global
        ))

        # Add randomness to cognitives and socials
        r1 = np.random.rand(*self.positions.shape)
        r2 = np.random.rand(*self.positions.shape)

        # Vectorized cognitives, socials and velocity updates
        cognitives = self.c1s * r1 * (self.pbest_pos - self.positions)
        socials = self.c2s * r2 * (social_best - self.positions)
        self.velocities = self.inertias * self.velocities + cognitives + socials

        # Move particles and retrict the movement within bounds with clipping
        self.positions = np.clip(self.positions + self.velocities, self.lower_bounds, self.upper_bounds)

        # Evaluate again all the solutions / new positions
        self.values = self.objective(self.positions)

        # Update best personal solutions
        mask = self.values < self.pbest_val
        self.pbest_val[mask] = self.values[mask]
        self.pbest_pos[mask] = self.positions[mask]

        if gen != 0 and gen % 1 == 0:
            # Recalculate swarm diversity and reseed particles if necessary
            self.check_and_reseed_worst_particles()
