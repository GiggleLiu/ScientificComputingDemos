# Spinglass

The spin-glass model is a model of a disordered magnet, where the spins coupled to each other in a random way.
Let $G=(V,E)$ be a graph, where $V$ is the set of vertices and $E$ is the set of edges. The Hamiltonian of the model is given by:

```math
H = \sum_{(i,j)\in E}J_{ij}\sigma_i\sigma_j
```
where $J_{ij}$ are random variables, $\sigma_i\in\{-1,1\}$ are the spins, and the sum is over all edges of the graph.

This demo contains two examples:
1. Generic tensor network contraction for the solution space properties of spin-glass problem on a 3-regular graph.
2. Simulated annealing for the solution space properties of spin-glass problem on a square lattice.