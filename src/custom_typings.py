from typing import List, Tuple, Callable

Vector = List[float]
Fitness = float
Bounds = Tuple[float, float]
ObjectiveFunction = Callable[[Vector], Fitness]