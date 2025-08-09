from equations import *
from multi_swarm_optimizer import MultiSwarmOptimizer

# Parámetros del problema
my_function = ackley
bounds, dimension, stop_optimum_value = get_bounds_dimensions_and_stop_global_optimum(my_function, "hard")
bounds = [bounds] * dimension

# Instanciar el optimizador multi-enjambre
optimizer = MultiSwarmOptimizer(
    objective=my_function,
    dim=dimension,
    bounds=bounds,
    num_swarms=5,
    swarm_size=4000,
    top_social_bests_to_choose_per_swarm=10,
    c1_range_per_swarm=(0.5, 1.5),
    c2_range_per_swarm=(0.5, 1.5),
    inertia_range_per_swarm=(0.3, 0.8),
    no_improve_stop=None,
    max_generations=1500,
    migration_rate=0.005,
    migration_interval=20,
    adam_interval=100,
    shrink_rounds=5,
    stop_score=stop_optimum_value,
    top_k_for_next_round_bounds=50,
    min_relative_variance_before_reseeding=0.001
)

# Ejecutar optimización
best_solution, best_value = optimizer.optimize()

print("Mejor solución encontrada:", best_solution)
print("Valor de la función objetivo:", best_value)
