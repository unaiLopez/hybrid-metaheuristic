# Hybrid Multi-Swarm Particle Optimization with Adam and Adaptive Bounds (HMS-POAAB)

**Hybrid Multi-Swarm Particle Optimization with Adam and Adaptive Bounds**

HMS-POAAB is a hybrid metaheuristic optimization algorithm that I built from scratch through extensive experimentation — including trying out a bunch of ideas that didn’t work nearly as well. It’s designed to solve difficult continuous-variable minimization problems, especially in **high-dimensional, rugged search spaces** filled with local minima.

The algorithm combines:

* **Multi-Swarm Particle Swarm Optimization (PSO)** — multiple swarms explore independently, periodically exchanging particles to increase exploration and avoid stagnation.
* **Adaptive Bound Shrinkage** — think of it like “zooming in” on the search space every some iterations (optimization round), progressively narrowing the bounds to focus on promising regions.
* **Adam-based Local Refinement** — randomly selected particles undergo a local gradient-based search using Adam optimization, with numerical gradients computed via central finite differences.
* **Diversity-based Reinitialization** — if the swarm diversity drops too low, the algorithm reinitializes a fraction of the worst-performing particles to restore exploration.

---

## ✨ Key Features

* **Multi-swarm independence** with periodic migration for both exploration and exploitation.
* **Per-particle cognitive and social coefficients** for increased behavioral diversity.
* **Adam refinement step** to help particles escape local minima.
* **Diversity monitoring** to detect premature convergence and reinitialize part of the swarm.
* **Iterative bound shrinking** to reduce search space complexity in large-scale problems.

---

## 📈 Benchmarking

HMS-POAAB was tested on a benchmark inspired by the paper:
[Benchmarking of Optimization Algorithms with Complex Functions](https://joiv.org/index.php/joiv/article/download/65/66?__cf_chl_tk=DyQSbMtVJAe8EIj0QOMqO69ERZ_Ix2yflVFE5nApvvk-1753531352-1.0.1.1-F8vIOG593HhB3rxEJvJHktpM2WYw5d2beBPerMP2nV8)

This benchmark includes:

* Highly multi-modal functions (many local minima)
* Both low- and high-dimensional problem instances
* Convex and non-convex landscapes

HMS-POAAB consistently found near-global optima quickly, even on **hundreds-dimensional** functions that are notoriously difficult to optimize.

---

## 🚀 Getting Started

### Requirements

```bash
python >= 3.8
numpy
```

### Example Usage

```python
from multi_swarm_optimizer import MultiSwarmOptimizer
from equations import ackley

# Define problem
dim = 20
bounds = [(-32.0, 32.0)] * dim
objective = ackley  # must return numpy array of values for given positions

# Create optimizer
optimizer = MultiSwarmOptimizer(
    objective=objective,
    dim=dim,
    bounds=bounds,
    swarm_size=50,
    num_swarms=5
)

# Run optimization
best_solution, best_value = optimizer.optimize()
print("Best solution:", best_solution)
print("Best value:", best_value)
```

---

## ⚠️ Current Limitations

* Works only for **continuous-variable minimization problems**
* Extension to discrete or maximization problems is possible, but not implemented yet
* Written for experimentation — not yet fully optimized for speed but most of the code is vectorized using numpy

---

## 📬 Contributing

Feel free to fork this repo, run it on your own problems, and share results!
I’d love to hear feedback, suggestions, or see improvements from the community.

---

## 📜 License

MIT License — free to use, modify, and distribute.

---
