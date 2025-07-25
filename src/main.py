from equations import *
from multi_swarm_optimizer import MultiSwarmOptimizer

# Parámetros del problema
bounds, dimension = get_bounds_and_dimensions(schwefel_2_22)
print(f"DIMENSION {dimension}")
print(f"BOUNDS {bounds}")

# Instanciar el optimizador multi-enjambre
optimizer = MultiSwarmOptimizer(
    objective=schwefel_2_22,
    dim=400,
    bounds=bounds,
    num_swarms=5,
    swarm_size=2000,
    max_generations=10000,
    migration_rate=0.2,
)

# Ejecutar optimización
best_solution, best_value = optimizer.optimize()

print("Mejor solución encontrada:", best_solution)
print("Valor de la función objetivo:", best_value)
