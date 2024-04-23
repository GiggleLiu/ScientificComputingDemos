# Spinglass

The spin-glass model is a model of a disordered magnet, where the spins coupled to each other in a random way.
Let $G=(V,E)$ be a graph, where $V$ is the set of vertices and $E$ is the set of edges. The Hamiltonian of the model is given by:

```math
H = \sum_{(i,j)\in E}J_{ij}\sigma_i\sigma_j + \sum_{i\in V}h_i\sigma_i,
```
where $J_{ij}$ and $h_i$ are the coupling constants and the external fields, respectively, $\sigma_i\in\{-1,1\}$ are the spins, and the sum is over all edges of the graph.

## Contents
Consider the problem of solving the ground state of the spin-glass model on a 3-regular graph. We provide the following methods to solve the problem:
1. Generic tensor network contraction that suited for obtaining solution space properties.[^Liu2021][^Liu2023]
2. Simulated annealing for the solution space properties of spin-glass problem on a fully connected graph.[^Cain2023][^SSSS]
3. Reducing the spin-glass problem to the circuit satisfiability problem.[^Nguyen2023][^Glover2019]

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificCompuingDemos
$ make init-Spinglass   # initialize the environment in Spinglass and Spinglass/examples
$ make example-Spinglass   # run the script Spinglass/examples/main.jl
```

## References
[^Liu2021]: Liu, Jin-Guo, Lei Wang, and Pan Zhang. [Tropical tensor network for ground states of spin glasses.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.090506) Physical Review Letters 126.9 (2021): 090506.
[^Liu2023]: Liu, Jin-Guo, et al. [Computing solution space properties of combinatorial optimization problems via generic tensor networks.](https://epubs.siam.org/doi/abs/10.1137/22M1501787) SIAM Journal on Scientific Computing 45.3 (2023): A1239-A1270.
[^SSSS]: Deep Learning and Quantum Programming: A Spring School, https://github.com/QuantumBFS/SSSS
[^Cain2023]: Cain, M., et al. "Quantum speedup for combinatorial optimization with flat energy landscapes (2023)." arXiv preprint arXiv:2306.13123.
[^Nguyen2023]: Nguyen, Minh-Thi, et al. "Quantum optimization with arbitrary connectivity using Rydberg atom arrays." PRX Quantum 4.1 (2023): 010316.
[^Glover2019]: Glover, Fred, Gary Kochenberger, and Yu Du. "Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models." 4or 17.4 (2019): 335-371.