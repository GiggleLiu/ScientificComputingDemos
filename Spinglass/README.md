# Spinglass

The spin-glass model is a model of a disordered magnet, where the spins are coupled to each other in a random way.
Let $G=(V,E)$ be a graph, where $V$ is the set of vertices and $E$ is the set of edges. The Hamiltonian of the model is given by:

```math
H = \sum_{(i,j)\in E}J_{ij}\sigma_i\sigma_j + \sum_{i\in V}h_i\sigma_i,
```
where $J_{ij}$ and $h_i$ are the coupling constants and the external fields, respectively, $\sigma_i\in\{-1,1\}$ are the spins, and the sum is over all edges of the graph.

## Contents

This repository focuses on solving the ground state of the spin-glass model on a 3-regular graph. We provide the following methods:

1. Reducing a circuit satisfiability problem to a spin-glass problem.[^Nguyen2023][^Glover2019]
2. Simulated annealing for solving a spin-glass ground state problem.[^Cain2023][^SSSS]

## To run

Clone the repository to your local machine and install the required packages (in a terminal):

```bash
$ git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
$ cd ScientificComputingDemos
$ dir=Spinglass make init   # initialize the environment in Spinglass and Spinglass/examples
$ dir=Spinglass make example   # run the script Spinglass/examples/main.jl
```

## To run the CUDA example

```bash
$ julia --project=Spinglass/examples Spinglass/examples/cuda.jl
```

## References
[^SSSS]: Deep Learning and Quantum Programming: A Spring School, https://github.com/QuantumBFS/SSSS
[^Cain2023]: Cain, M., et al. "Quantum speedup for combinatorial optimization with flat energy landscapes (2023)." arXiv preprint arXiv:2306.13123.
[^Nguyen2023]: Nguyen, Minh-Thi, et al. "Quantum optimization with arbitrary connectivity using Rydberg atom arrays." PRX Quantum 4.1 (2023): 010316.
[^Glover2019]: Glover, Fred, Gary Kochenberger, and Yu Du. "Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models." 4or 17.4 (2019): 335-371.
