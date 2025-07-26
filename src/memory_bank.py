import numpy as np

class MemoryBank:
    def __init__(self):
        self.solutions = []

    def update(self, solution: np.ndarray, value: float) -> None:
        self.solutions.append((solution.copy(), value))
        self.solutions.sort(key=lambda x: x[1])
        self.solutions = self.solutions[:20]

    def get_top_k(self, k: int = 5) -> list:
        return [sol for sol, _ in self.solutions[:k]]