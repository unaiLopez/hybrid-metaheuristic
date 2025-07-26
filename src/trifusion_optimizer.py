from multi_swarm_optimizer import MultiSwarmOptimizer
from typing import Callable, Tuple
import numpy as np

class TrifusionOptimizer:
    def __init__(
            self,
            objective: Callable[[np.ndarray], np.ndarray],
            dim: int,
            bounds: Tuple[float, float],
            num_shrinkage_iterations: int = 3,
            bounds_shrinkage_factor: float = 0.75,
            num_swarms: int = 3,
            swarm_size: int = 20,
            max_generations: int = 100,
            num_generations_no_improve_stop: int = 20,
            stop_score: float = 1e-6,
            migration_rate: float = 0.1,
    ):

        self.objective = objective
        self.dim = dim
        self.bounds = bounds
        self.num_shrinkage_iterations = num_shrinkage_iterations
        self.bounds_shrinkage_factor = bounds_shrinkage_factor
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.max_generations = max_generations
        self.num_generations_no_improve_stop = num_generations_no_improve_stop
        self.stop_score = stop_score
        self.migration_rate = migration_rate

        self.best_global_value = float('inf')
        self.best_global_solution = None

    def optimize(self) -> Tuple[np.ndarray, float]:
        bounds = self.bounds
        original_lower_bounds = np.array([b[0] for b in bounds])
        original_upper_bounds = np.array([b[1] for b in bounds])

        lower_bounds = original_lower_bounds.copy()
        upper_bounds = original_upper_bounds.copy()

        for i in range(self.num_shrinkage_iterations):
            if i != 0:
                # Calcula un peso progresivo para confiar más en el best point con cada iteración
                w = i / self.num_shrinkage_iterations  # entre 0.0 y 1.0

                dim_widths = (upper_bounds - lower_bounds) * self.bounds_shrinkage_factor
                best = self.best_global_solution

                centered_lower = best - dim_widths / 2
                centered_upper = best + dim_widths / 2

                # Mezcla ponderada entre zona centrada y los límites originales
                new_lower_bounds = w * centered_lower + (1 - w) * original_lower_bounds
                new_upper_bounds = w * centered_upper + (1 - w) * original_upper_bounds

                # Asegura que estén dentro de los límites originales
                new_lower_bounds = np.maximum(new_lower_bounds, original_lower_bounds)
                new_upper_bounds = np.minimum(new_upper_bounds, original_upper_bounds)

                bounds = list(zip(new_lower_bounds, new_upper_bounds))
                lower_bounds = new_lower_bounds
                upper_bounds = new_upper_bounds

            multiswarm_optimizer = MultiSwarmOptimizer(
                objective=self.objective,
                dim=self.dim,
                bounds=bounds,
                num_swarms=self.num_swarms,
                max_generations=self.max_generations,
                num_generations_no_improve_stop=self.num_generations_no_improve_stop,
                stop_score=self.stop_score,
                migration_rate=self.migration_rate
            )
            best_solution, best_value = multiswarm_optimizer.optimize()

            if best_value < self.best_global_value:
                self.best_global_value = best_value
                self.best_global_solution = best_solution

            if best_value <= self.stop_score:
                break
        
        print("ULTIMOS BOUNDS")
        print(bounds)
        return self.best_global_solution, self.best_global_value


from equations import *

my_function = ackley
# Parámetros del problema
bounds, dimension, stop_optimum_value = get_bounds_dimensions_and_stop_global_optimum(my_function, "hard")
#dimension = 120
#bounds = [(-1.0, 1.9), (-2.12, 5.12), (-1.0, 3.9), (-1.0, 312.1)] * 30
bounds = [bounds] * dimension

# Instanciar el optimizador Trifusion
optimizer = TrifusionOptimizer(
    objective=my_function,
    dim=dimension,
    bounds=bounds,
    num_shrinkage_iterations=5,
    bounds_shrinkage_factor=0.75,
    num_swarms=5,
    swarm_size=20_000,
    num_generations_no_improve_stop=200,
    max_generations=200_000,
    migration_rate=0.2,
    stop_score=stop_optimum_value
)

# Ejecutar optimización
best_solution, best_value = optimizer.optimize()

print("Mejor solución encontrada:", best_solution)
print("Valor de la función objetivo:", best_value)


