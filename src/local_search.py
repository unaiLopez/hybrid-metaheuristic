import random

from typing import Tuple
from custom_typings import ObjectiveFunction, Vector, Bounds, Fitness

def ballistic_local_search(center: Vector, objective: ObjectiveFunction, bounds: Bounds, depth: int = 10, spread: float = 1.0) -> Tuple[Vector, Fitness]:
    best = center[:]
    best_score = objective(best)
    
    for _ in range(depth):
        candidate = []
        for x in center:
            # Genera un valor candidato y lo limita dentro de bounds
            val = x + random.uniform(-spread, spread)
            val = max(bounds[0], min(bounds[1], val))
            candidate.append(val)
        
        score = objective(candidate)
        if score < best_score:
            best = candidate
            best_score = score

    return best, best_score