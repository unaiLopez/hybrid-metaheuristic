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
        max_generations: int,
        migration_rate: float,
        num_generations_no_improve_stop: int
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
                    max_generations=max_generations,
                    migration_rate=migration_rate,
                    stop_score=stop_global_optimum,
                    num_generations_no_improve_stop=num_generations_no_improve_stop
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
        sphere,
        rosenbrock,
        quartic,
        step,
        schwefel_2_22,
        sum_square,
        elliptic,
        griewank,
        ackley,
        weierstrass,
        non_continuous_rastrigin,
        penalized2,
        schaffer,
        alpine,
        himmelblau
    ]

    benchmark = Benchmark(funcs)
    summary = benchmark.run_benchmark(
        runs=1,
        mode="hard",
        num_swarms=10,
        swarm_size=2_000,
        max_generations=35000,
        migration_rate=0.2,
        num_generations_no_improve_stop=500
    )

    # Print summary
    for fname, stats in summary.items():
        print(f"Function: {fname}")
        print(f"  All Runs:   {stats['runs']}")
        print(f"  All Execution Times:   {stats['execution_times']}")
        print(f"  All Solutions:   {stats['best_solutions']}")
        print(f"  Mean Execution Times:   {stats['mean_execution_times']:.4f}")
        print(f"  Mean Values:   {stats['mean_val']:.4f}")
        print(f"  Median Values: {stats['median_val']:.4f}")
        print(f"  Min Values:    {stats['min_val']:.4f}")
        print(f"  Max Values:    {stats['max_val']:.4f}\n")
