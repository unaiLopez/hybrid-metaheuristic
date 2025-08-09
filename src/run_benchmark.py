import os
import csv
import time
import numpy as np

from equations import *
from multi_swarm_optimizer import MultiSwarmOptimizer
from typing import List, Dict, Any

class Benchmark:
    def __init__(self, functions: List[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.functions = functions

    def run_benchmark(
        self,
        mode: str,
        runs: int,
        num_swarms: int,
        swarm_size: int,
        top_social_bests_to_choose_per_swarm: int,
        c1_range_per_swarm: Tuple[float, float],
        c2_range_per_swarm: Tuple[float, float],
        inertia_range_per_swarm: Tuple[float, float],
        max_generations: int,
        migration_rate: float,
        no_improve_stop: int,
        migration_interval: int,
        adam_interval: int,
        shrink_rounds: int,
        top_k_for_next_round_bounds: int,
        min_relative_variance_before_reseeding: float
    ) -> Dict[str, Dict[str, Any]]:
        
        results: Dict[str, Dict[str, Any]] = {}

        for func in self.functions:
            name = func.__name__
            bounds, dim, stop_global_optimum = get_bounds_dimensions_and_stop_global_optimum(func, mode)
            bounds = [bounds] * dim

            values: List[float] = []
            solutions: List[np.ndarray] = []
            times: List[float] = []
            for _ in range(runs):
                start_time = time.time()
                optimizer = MultiSwarmOptimizer(
                    objective=func,
                    dim=dim,
                    bounds=bounds,
                    num_swarms=num_swarms,
                    swarm_size=swarm_size,
                    top_social_bests_to_choose_per_swarm=top_social_bests_to_choose_per_swarm,
                    c1_range_per_swarm=c1_range_per_swarm,
                    c2_range_per_swarm=c2_range_per_swarm,
                    inertia_range_per_swarm=inertia_range_per_swarm,
                    max_generations=max_generations,
                    migration_rate=migration_rate,
                    stop_score=stop_global_optimum,
                    no_improve_stop=no_improve_stop,
                    migration_interval=migration_interval,
                    adam_interval=adam_interval,
                    shrink_rounds=shrink_rounds,
                    top_k_for_next_round_bounds=top_k_for_next_round_bounds,
                    min_relative_variance_before_reseeding=min_relative_variance_before_reseeding
                )
                best_sol, best_val = optimizer.optimize()
                values.append(best_val)
                solutions.append(best_sol)
                times.append(time.time() - start_time)

            # Compute statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            min_val = min(values)
            max_val = max(values)

            results[name] = {
                'runs': [float(value) for value in values],
                'bounds': bounds[0],
                'dimensions': dim,
                'execution_times': times,
                'best_solutions': solutions,
                'mean_execution_times': np.mean(times),
                'mean_val': mean_val,
                'median_val': median_val,
                'min_val': min_val,
                'max_val': max_val,
            }

        return results

if __name__ == '__main__':
    funcs = [
        weierstrass,
        sphere,
        rosenbrock,
        quartic,
        step,
        schwefel_2_22,
        sum_square,
        elliptic,
        griewank,
        ackley,
        non_continuous_rastrigin,
        penalized2,
        schaffer,
        alpine,
        himmelblau
    ]
    mode = "hard"
    benchmark = Benchmark(funcs)
    summary = benchmark.run_benchmark(
        runs=5,
        mode=mode,
        num_swarms=5,
        swarm_size=2_000,
        top_social_bests_to_choose_per_swarm=10,
        c1_range_per_swarm=(1.0, 1.5),
        c2_range_per_swarm=(1.0, 1.5),
        inertia_range_per_swarm=(0.4, 0.8),
        max_generations=2000,
        migration_rate=0.01,
        no_improve_stop=None,
        migration_interval=100,
        adam_interval=200,
        shrink_rounds=3,
        top_k_for_next_round_bounds=50,
        min_relative_variance_before_reseeding=0.001
    )

    rows = []
    for fname, stats in summary.items():
        for run, execution_time in zip(stats['runs'], stats['execution_times']):
            rows.append({
                "Function": fname,
                "Bounds": stats['bounds'],
                "Dimensions": stats['dimensions'],
                "Mode": mode,
                "All Runs": run,
                "All Execution Times": execution_time,
                "Mean Execution Times": stats['mean_execution_times'],
                "Mean Values": stats['mean_val'],
                "Median Values": stats['median_val'],
                "Min Values": stats['min_val'],
                "Max Values": stats['max_val']
            })

    # Guardar en CSV
    os.makedirs("../output", exist_ok=True)
    with open(f"../output/benchmark_result_mode_{mode}.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Archivo 'benchmark_result_mode_{mode}.csv' guardado con éxito.")
