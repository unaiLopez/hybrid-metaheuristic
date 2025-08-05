import numpy as np

class MemoryBank:
    def __init__(self):
        self.solutions = []          # lista ordenada de (solución, valor)
        self.solution_set = set()    # set de llaves hashables para detección de duplicados

    def _hashable(self, solution: np.ndarray, value: float) -> tuple:
        # primero redondeamos el vector con np.round y lo convertimos a tupla de floats
        sol_key = tuple(np.round(solution, 8).tolist())
        # luego redondeamos el valor escalar
        val_key = round(float(value), 8)
        return (sol_key, val_key)

    def update(self, solution: np.ndarray, value: float) -> None:
        key = self._hashable(solution, value)
        if key in self.solution_set:
            return  # ya existe la misma (solución, valor)

        # añadimos y recortamos a top‐20
        self.solutions.append((solution.copy(), float(value)))
        self.solutions.sort(key=lambda x: x[1])
        self.solutions = self.solutions[:20]

        # reconstruimos el set para mantenerlo consistente
        self.solution_set = { self._hashable(sol, val) for sol, val in self.solutions }

    def get_top_k(self, k: int = 5) -> list:
        return [sol for sol, _ in self.solutions[:k]]
