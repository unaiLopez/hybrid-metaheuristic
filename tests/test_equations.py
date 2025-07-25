import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
import numpy as np
from equations import (
    sphere, rosenbrock, quartic, step, schwefel_2_22,
    sum_square, elliptic, rastrigin, griewank, ackley, weierstrass,
    non_continuous_rastrigin, penalized2, schaffer, alpine,
    himmelblau
)

# Define test settings: (function, min_dim, com_dim, bounds, optimum)
TEST_SETTINGS = [
    (sphere, 3, 30, (-100, 100), lambda d: np.zeros(d)),
    (rosenbrock, 5, 30, (-30, 30), lambda d: np.ones(d)),
    (quartic, 10, 30, (-1.28, 1.28), lambda d: np.zeros(d)),
    (step, 5, 30, (-100, 100), lambda d: -0.5 * np.ones(d)),
    (schwefel_2_22, 10, 30, (-10, 10), lambda d: np.zeros(d)),
    (sum_square, 10, 30, (-10, 10), lambda d: np.zeros(d)),
    (elliptic, 10, 30, (-100, 100), lambda d: np.zeros(d)),
    (rastrigin, 2, 30, (-5.12, 5.12), lambda d: np.zeros(d)),
    (griewank, 2, 30, (-600, 600), lambda d: np.zeros(d)),
    (ackley, 5, 30, (-32, 32), lambda d: np.zeros(d)),
    (weierstrass, 10, 30, (-0.5, 0.5), lambda d: np.zeros(d)),
    (non_continuous_rastrigin, 10, 30, (-5.12, 5.12), lambda d: np.zeros(d)),
    (penalized2, 10, 30, (-50, 50), lambda d: np.ones(d)),
    (schaffer, 2, 2, (-100, 100), lambda d: np.zeros(d)),
    (alpine, 10, 30, (-10, 10), lambda d: np.zeros(d)),
    (himmelblau, 2, 2, (-6, 6), lambda d: np.array([3.0, 2.0]))
]

@pytest.mark.parametrize("func, min_dim, com_dim, bounds, opt_fn", TEST_SETTINGS)
def test_minimum_value(func, min_dim, com_dim, bounds, opt_fn):
    """
    Test that each function returns its known minimum at optimum.
    """
    for dim in (min_dim, com_dim):
        x_opt = np.atleast_2d(opt_fn(dim))
        val = func(x_opt)
        assert val.shape == (1,)
        assert np.allclose(val, 0, atol=1e-4), f"{func.__name__} did not return 0 at optimum"

@pytest.mark.parametrize("func, min_dim, com_dim, bounds, _", TEST_SETTINGS)
def test_random_evaluation(func, min_dim, com_dim, bounds, _):
    """
    Test random inputs within bounds produce finite outputs and correct shape.
    """
    low, high = bounds
    for dim in (min_dim, com_dim):
        X = np.random.uniform(low, high, size=(5, dim))
        vals = func(X)
        assert vals.shape == (5,)
        assert np.all(np.isfinite(vals)), f"{func.__name__} produced non-finite values"
