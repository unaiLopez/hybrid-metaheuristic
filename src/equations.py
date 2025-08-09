import numpy as np
from typing import Tuple, Callable

# Benchmark is based on the benchmark for metaheuristics proposed in this paper
# https://joiv.org/index.php/joiv/article/download/65/66?__cf_chl_tk=DyQSbMtVJAe8EIj0QOMqO69ERZ_Ix2yflVFE5nApvvk-1753531352-1.0.1.1-F8vIOG593HhB3rxEJvJHktpM2WYw5d2beBPerMP2nV8

def sphere(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0].
    """
    x = np.atleast_2d(x)
    return np.sum(x**2, axis=1)

def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[1,1,…,1]
    """

    x = np.atleast_2d(x)
    xi = x[:, :-1]
    xi_plus1 = x[:, 1:]
    return np.sum(100.0 * (xi_plus1 - xi**2)**2 + (1 - xi)**2, axis=1)

def quartic(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0]
    """

    x = np.atleast_2d(x)
    return np.sum(x**4, axis=1)

def step(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[-0.5,-0.5,…,-0.5]
    """

    x = np.atleast_2d(x)
    return np.sum((x + 0.5)**2, axis=1)

def schwefel_2_22(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0]
    """

    x = np.atleast_2d(x)
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)


def sum_square(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0]
    """

    x = np.atleast_2d(x)
    n = x.shape[1]
    indices = np.arange(1, n + 1)
    return np.sum(indices * x**2, axis=1)

def elliptic(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0]
    """

    x = np.atleast_2d(x)
    n = x.shape[1]
    exponents = np.arange(n) / (n - 1)
    coefficients = 10**6 ** exponents
    return np.sum(coefficients * x**2, axis=1)

def rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0]
    """

    x = np.atleast_2d(x)
    n = x.shape[1]
    indices = np.arange(1, n + 1)
    return np.sum((x**2) - 10 * np.cos(2 * np.pi * indices) + 10, axis=1)

def griewank(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0]
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    indices = np.arange(1, n + 1)
    
    sum_term = np.sum(x**2, axis=1) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(indices)), axis=1)
    
    result = 1 + sum_term - prod_term
    return result

def ackley(x: np.ndarray) -> np.ndarray:
    """
    Minimum is 0 at f(x*)=[0,0,…,0]
    """

    x = np.atleast_2d(x)
    
    n = x.shape[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(c * x), axis=1)
    
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.e

def weierstrass(x: np.ndarray, a: float = 0.5, b: int = 3, k_max: int = 20) -> np.ndarray:
    """
    Global optimum at f(x*)=[0,0,...,0] in domain [-0.5,0.5]
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    
    k = np.arange(k_max + 1)
    a_k = a**k
    b_k = b**k
    
    main_sum = 0
    for i in range(n):
        main_sum += np.sum(a_k * np.cos(2 * np.pi * b_k * x[:, i:i+1]), axis=1)
    
    correction = n * np.sum(a_k * np.cos(2 * np.pi * b_k * 0))
    
    return main_sum - correction

def non_continuous_rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Global minimum at f(x*)=[0,0,...,0] in domain [-5.12,5.12]
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    
    y = np.where(np.abs(x) < 0.5, x, np.round(2 * x) / 2)
    
    return 10 * n + np.sum(y**2 - 10 * np.cos(2 * np.pi * y), axis=1)

def penalized2(x: np.ndarray, a: float = 5, k: int = 100, m: int = 4) -> np.ndarray:
    """
    Global minimum at f(x*)=[1,1,...,1] in domain [-50,50]
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    
    sum_term = np.sin(3 * np.pi * x[:, 0])**2
    if n > 1:
        middle_terms = np.sum((x[:, :-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[:, 1:])**2), axis=1)
        sum_term += middle_terms
    sum_term += (x[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * x[:, -1])**2)
    
    def u_penalty(xi, a, k, m):
        return np.where(xi > a, k * (xi - a)**m,
                       np.where(xi < -a, k * (-xi - a)**m, 0))
    
    penalty = np.sum(u_penalty(x, a, k, m), axis=1)
    
    return 0.1 * sum_term + penalty

def schaffer(x: np.ndarray) -> np.ndarray:
    """
    Global minimum at f(x*)=[0,0,...,0] in domain [-100,100]
    """
    x = np.atleast_2d(x)
    
    numerator = np.sin(np.sqrt(np.sum(x**2, axis=1)))**2 - 0.5
    denominator = (1 + 0.001 * np.sum(x**2, axis=1))**2
    
    return 0.5 + numerator / denominator

def alpine(x: np.ndarray) -> np.ndarray:
    """
    Global minimum at f(x*)=[0,0,...,0] in domain [-10,10]
    """
    x = np.atleast_2d(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)

def himmelblau(x: np.ndarray) -> np.ndarray:
    """
    Multiple global minima at four locations in domain [-6,6]
    f(x*)=0 at (3,2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    """
    x = np.atleast_2d(x)
    x1, x2 = x[:, 0], x[:, 1]
    
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def get_bounds_dimensions_and_stop_global_optimum(func: Callable, mode: str) -> Tuple[Tuple[float, float], int]:
    name = func.__name__

    if mode == "easy":
        lookup = {
            "weierstrass":            ((-0.5, 0.5), 10, -39.99),
            "sphere":                 ((-5.12, 5.12), 3, 1e-6),
            "rosenbrock":             ((-5, 10), 5, 1e-6),
            "quartic":                ((-1.28, 1.28), 10, 1e-6),
            "step":                   ((-100, 100), 5, 1e-6),
            "schwefel_2_22":          ((-10, 10), 10, 1e-6),
            "sum_square":            ((-10, 10), 10, 1e-6),
            "elliptic":               ((-100, 100), 10, 1e-6),
            "ackley":                 ((-32, 32), 5, 1e-6),
            "griewank":               ((-600, 600), 2, 1e-6),
            "rastrigin":              ((-5.12, 5.12), 10, 1e-6),
            "non_continuous_rastrigin": ((-5.12, 5.12), 10, 1e-6),
            "penalized2":             ((-5.12, 5.12), 10, 1e-6),
            "schaffer":               ((-100, 100), 2, 1e-6),
            "alpine":                 ((-10, 10), 10, 1e-6),
            "himmelblau":             ((-6, 6), 30, 1e-6)
        }
    elif mode == "medium":
        lookup = {
            "weierstrass":            ((-0.5, 0.5), 30, -119.99),
            "sphere":                 ((-5.12, 5.12), 30, 1e-6),
            "rosenbrock":             ((-5, 10), 30, 1e-6),
            "quartic":                ((-1.28, 1.28), 30, 1e-6),
            "step":                   ((-100, 100), 30, 1e-6),
            "schwefel_2_22":          ((-10, 10), 30, 1e-6),
            "sum_square":            ((-10, 10), 30, 1e-6),
            "elliptic":               ((-100, 100), 30, 1e-6),
            "ackley":                 ((-32, 32), 30, 1e-6),
            "griewank":               ((-600, 600), 30, 1e-6),
            "rastrigin":              ((-5.12, 5.12), 30, 1e-6),
            "non_continuous_rastrigin": ((-5.12, 5.12), 30, 1e-6),
            "penalized2":             ((-5.12, 5.12), 30, 1e-6),
            "schaffer":               ((-100, 100), 2, 1e-6),
            "alpine":                 ((-10, 10), 30, 1e-6),
            "himmelblau":             ((-6, 6), 50, 1e-6)
        }
    elif mode == "hard":
        lookup = {
            "weierstrass":            ((-0.5, 0.5), 100, -399.99),
            "sphere":                 ((-5.12, 5.12), 256, 1e-6),
            "rosenbrock":             ((-5, 10), 100, 1e-6),
            "quartic":                ((-1.28, 1.28), 100, 1e-6),
            "step":                   ((-100, 100), 100, 1e-6),
            "schwefel_2_22":          ((-10, 10), 100, 1e-6),
            "sum_square":            ((-10, 10), 100, 1e-6),
            "elliptic":               ((-100, 100), 100, 1e-6),
            "ackley":                 ((-32, 32), 128, 1e-6),
            "griewank":               ((-600, 600), 100, 1e-6),
            "rastrigin":              ((-5.12, 5.12), 100, 1e-6),
            "non_continuous_rastrigin": ((-5.12, 5.12), 100, 1e-6),
            "penalized2":             ((-5.12, 5.12), 100, 1e-6),
            "schaffer":               ((-100, 100), 50, 1e-6),
            "alpine":                 ((-10, 10), 100, 1e-6),
            "himmelblau":             ((-6, 6), 200, 1e-6)
        }
    
    return lookup.get(name)