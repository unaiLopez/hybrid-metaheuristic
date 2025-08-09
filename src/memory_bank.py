import numpy as np

class MemoryBank:
    def __init__(self, top_solutions_to_save: int = 100):
        self.solutions = []
        self.solution_set = set()
        self.top_solutions_to_save = top_solutions_to_save

    def _hashable(self, solution: np.ndarray, value: float) -> tuple:
        sol_key = tuple(np.round(solution, 8).tolist())
        val_key = round(float(value), 8)
        return (sol_key, val_key)

    def update(self, solution: np.ndarray, value: float) -> None:
        key = self._hashable(solution, value)
        if key in self.solution_set:
            return

        self.solutions.append((solution.copy(), float(value)))
        self.solutions.sort(key=lambda x: x[1])
        self.solutions = self.solutions[:self.top_solutions_to_save]

        self.solution_set = {self._hashable(sol, val) for sol, val in self.solutions}

    def get_top_k(self, k) -> list:
        return [sol for sol, _ in self.solutions[:k]]
