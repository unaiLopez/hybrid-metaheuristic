import numpy as np
from typing import Callable, Tuple

def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-8) -> np.ndarray:
    fx = f(x)[0]
    grad = np.zeros_like(x)

    # Perturbar una dimensión a la vez
    perturb = np.eye(len(x)) * h  # matriz identidad escalada
    xs_perturbed = x + perturb    # (n, n)
    
    fxs = np.array([float(f(xi)) for xi in xs_perturbed])  # aseguramos escalar por evaluación
    grad = (fxs - fx) / h
    return grad

def adam_optimize(
    f: Callable[[np.ndarray], float],
    x_init: np.ndarray,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    
    x = x_init.copy()
    m = np.zeros_like(x)  # 1st moment vector
    v = np.zeros_like(x)  # 2nd moment vector

    for t in range(1, max_iters + 1):
        grad = numerical_gradient(f, x)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        x -= update

        if np.linalg.norm(grad) < tol:
            print(f"Converged at iteration {t}")
            break

        if np.linalg.norm(update) < tol:
            print(f"[Adam] Update too small — stopping at iteration {t}")
            break

    return x, f(x)[0]
