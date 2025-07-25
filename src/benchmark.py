import numpy as np

from equations import *
from multi_swarm_optimizer import MultiSwarmOptimizer
from typing import List, Dict, Any
from custom_typings import ObjectiveFunction


class Benchmark:
    def __init__(self, functions: List[ObjectiveFunction]) -> None:
        self.functions = functions

    def apply_benchmark(
        self,
        runs,
        num_swarms: int,
        swarm_size: int,
        max_generations: int,
        migration_rate: float
    ) -> Dict[str, Dict[str, Any]]:
        
        results: Dict[str, Dict[str, Any]] = {}

        for func in self.functions:
            name = func.__name__
            bounds, dim = get_bounds_and_dimensions(func)

            values: List[float] = []
            for _ in range(runs):
                # Run optimization
                optimizer = MultiSwarmOptimizer(
                    objective=func,
                    dim=dim,
                    bounds=bounds,
                    num_swarms=num_swarms,
                    swarm_size=swarm_size,
                    max_generations=max_generations,
                    migration_rate=migration_rate,
                )
                _, best_val = optimizer.optimize()
                values.append(best_val)

            # Compute statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            min_val = min(values)
            max_val = max(values)

            results[name] = {
                'runs': [float(value) for value in values],
                'mean': mean_val,
                'median': median_val,
                'min': min_val,
                'max': max_val,
            }

        return results

if __name__ == '__main__':
    funcs = [
        #sphere,
        #rosenbrock,
        #quartic,
        #step,
        #schwefel_2_22,
        #sum_square,
        elliptic,
        griewank,
        ackley,
        #weierstrass,
        #non_continuous_rastrigin,
        #penalized2,
        #schaffer,
        #alpine,
        #himmelblau
    ]
    bounds_dims = [get_bounds_and_dimensions(func) for func in funcs]

    bench = Benchmark(funcs)
    summary = bench.apply_benchmark(
        runs=5,
        num_swarms=20,
        swarm_size=10_000,
        max_generations=500,
        migration_rate=0.2
    )

    # Print summary
    for fname, stats in summary.items():
        print(f"Function: {fname}")
        print(f"  All Runs:   {stats['runs']}")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  Max:    {stats['max']:.4f}\n")
