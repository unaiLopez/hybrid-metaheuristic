import numpy as np
from typing import Callable, Tuple

def numerical_gradient_central_differentiation(objective: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-8) -> np.ndarray:
    dims = x.shape[0]

    I = np.eye(dims)
    X_expanded = x[np.newaxis, :]
    X_plus = X_expanded + h * I
    X_minus = X_expanded - h * I

    return (objective(X_plus) - objective(X_minus)) / (2 * h)

def adam_optimize(
    objective: Callable[[np.ndarray], float],
    x_init: np.ndarray,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    
    x = x_init.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, max_iters + 1):
        grad = numerical_gradient_central_differentiation(objective, x)

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

    return x, objective(x)[0]
