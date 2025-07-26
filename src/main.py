from equations import *
from multi_swarm_optimizer import MultiSwarmOptimizer

# Parámetros del problema
my_function = rastrigin
bounds, dimension, stop_optimum_value = get_bounds_dimensions_and_stop_global_optimum(my_function, "hard")
bounds = [bounds] * dimension

# Instanciar el optimizador multi-enjambre
optimizer = MultiSwarmOptimizer(
    objective=my_function,
    dim=dimension,
    bounds=bounds,
    num_swarms=5,
    swarm_size=2000,
    num_generations_no_improve_stop=1000,
    max_generations=10000,
    migration_rate=0.2,
    stop_score=stop_optimum_value
)

# Ejecutar optimización
best_solution, best_value = optimizer.optimize()

print("Mejor solución encontrada:", best_solution)
print("Valor de la función objetivo:", best_value)
