import random
import numpy as np
import local_search as local_search

from typing import Tuple
from memory_bank import MemoryBank
from custom_typings import ObjectiveFunction

class Swarm:
    def __init__(
        self,
        objective: ObjectiveFunction,
        dim: int,
        bounds: Tuple[float, float],
        swarm_size: int
    ) -> None:
        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.memory = MemoryBank()

        self.positions = np.random.uniform(bounds[0], bounds[1], size=(swarm_size, dim))
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
        social_best = (
            random.choice(self.memory.get_top_k(10))
            if self.memory.solutions else self.best_global
        )

        # Hiperparámetros PSO
        c1 = np.random.uniform(1.0, 2.5)
        c2 = np.random.uniform(1.0, 2.5)
        inertia = np.random.uniform(0.3, 0.9)
        r1 = np.random.rand(*self.positions.shape)
        r2 = np.random.rand(*self.positions.shape)

        # Actualización vectorizada
        cognitive = c1 * r1 * (self.pbest_pos - self.positions)
        social = c2 * r2 * (np.array(social_best) - self.positions)
        self.velocities = inertia * self.velocities + cognitive + social

        # Movimiento y restricción al dominio
        self.positions = np.clip(self.positions + self.velocities, self.bounds[0], self.bounds[1])

        # Reevaluar las posiciones
        self.values = self.objective(self.positions)

        # Actualización de mejores personales
        mask = self.values < self.pbest_val
        self.pbest_val[mask] = self.values[mask]
        self.pbest_pos[mask] = self.positions[mask]