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
        swarm_size: int
    ):
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])
        self.memory = MemoryBank()
        self.positions = np.array([np.random.uniform(low, high, size=swarm_size) for (low, high) in bounds]).T
        self.velocities = np.random.uniform(-1.0, 1.0, size=(swarm_size, dim))

        self.values = self.objective(self.positions)  # Evaluar todas las partículas
        self.pbest_pos = self.positions.copy()        # Mejores personales
        self.pbest_val = self.values.copy()

        self.best_index = np.argmin(self.values)
        self.best_global = self.positions[self.best_index].copy()
        self.best_score = self.values[self.best_index]
        self.memory.update(self.best_global, self.best_score)

    def step(self) -> None:
        # Actualizar el mejor global
        self.best_index = np.argmin(self.values)
        self.best_global = self.positions[self.best_index].copy()
        self.best_score = self.values[self.best_index]
        self.memory.update(self.best_global, self.best_score)

        # Obtener una solución social de la memoria (exploración cooperativa)
        social_best = np.array((
            random.choice(self.memory.get_top_k(10))
            if self.memory.solutions else self.best_global
        ))

        # Hiperparámetros PSO
        c1 = np.random.uniform(1.0, 2.5)
        c2 = np.random.uniform(1.0, 2.5)
        inertia = np.random.uniform(0.3, 0.9)
        r1 = np.random.rand(*self.positions.shape)
        r2 = np.random.rand(*self.positions.shape)

        # Actualización vectorizada
        cognitive = c1 * r1 * (self.pbest_pos - self.positions)
        social = c2 * r2 * (social_best - self.positions)
        self.velocities = inertia * self.velocities + cognitive + social

        # Movimiento y restricción al dominio
        self.positions = np.clip(self.positions + self.velocities, self.lower_bounds, self.upper_bounds)

        # Reevaluar las posiciones
        self.values = self.objective(self.positions)

        # Actualización de mejores personales
        mask = self.values < self.pbest_val
        self.pbest_val[mask] = self.values[mask]
        self.pbest_pos[mask] = self.positions[mask]