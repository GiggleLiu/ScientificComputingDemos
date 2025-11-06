# Spinglass

A Julia package for solving spin glass optimization problems using various advanced methods including simulated annealing, quantum-inspired dynamics, and specialized algorithms for maximum independent set problems.

## Overview

The spin-glass model represents a disordered magnetic system where spins interact through random couplings. Given a graph $G=(V,E)$, the Hamiltonian is:

```math
H = \sum_{(i,j)\in E}J_{ij}\sigma_i\sigma_j + \sum_{i\in V}h_i\sigma_i,
```

where:
- $J_{ij}$ are the coupling constants between spins
- $h_i$ are external magnetic fields  
- $\sigma_i\in\{-1,1\}$ are the spin variables
- The sum is over all edges and vertices of the graph

Finding the ground state (minimum energy configuration) of this model is NP-hard and appears in numerous applications from physics to computer science.

## Features

### 1. **Simulated Annealing** (`simulated_annealing.jl`)
Classical optimization using Metropolis-Hastings Monte Carlo with efficient energy caching for O(1) energy difference calculations.

**Key capabilities:**
- Multiple independent runs for better solution quality
- Flexible temperature schedules (linear, geometric, logarithmic)
- CPU and GPU implementations (via CUDA extension)
- Optimized for dense spin glass problems

### 2. **Quantum-Inspired Dynamics** (`dynamics.jl`)
Continuous-time oscillator dynamics inspired by quantum annealing, simulating the transverse field Ising model.

**Key capabilities:**
- Coupled oscillator equations with transverse field
- Parallel trajectory evolution
- Sherrington-Kirkpatrick (SK) model generation
- Suitable for exploring quantum-classical algorithm comparisons

### 3. **Maximum Independent Set Solver** (`mis_sa.jl`)
Specialized simulated annealing for finding maximum independent sets on graphs, with applications to Rydberg atom arrays.

**Key capabilities:**
- Three types of moves: addition, removal, spin-exchange
- Unit disk graph construction for spatial problems
- Equilibration sampling for statistical physics studies
- Efficient for sparse graphs and lattice structures

### 4. **GPU Acceleration** (`ext/CUDAExt.jl`)
CUDA extension for massive parallelization of simulated annealing.

**Key capabilities:**
- Parallel execution of 100s of annealing runs
- Automatic CPU ↔ GPU data transfer
- Efficient for large problem sizes (N ≥ 100) and many runs

## Installation

This package is part of the ScientificComputingDemos repository. Clone and set up:

```bash
git clone https://github.com/GiggleLiu/ScientificComputingDemos.git
cd ScientificComputingDemos
dir=Spinglass make init   # Initialize environment
```

## Quick Start

### Basic Simulated Annealing

```julia
using Spinglass

# Load a spin glass problem from file
# File format: each line is "vertex1 vertex2 coupling_strength" (0-indexed)
prob = load_spinglass("data/example.txt")

# Configure annealing parameters
tempscales = collect(10.0:-0.15:0.55)  # Temperature schedule
nupdate = 4000                          # Updates per temperature  
nrun = 30                               # Independent runs

# Solve
opt_cost, opt_config = anneal(nrun, prob, tempscales, nupdate)
println("Optimal energy: ", opt_cost)
println("Configuration: ", opt_config.config)
```

### GPU-Accelerated Annealing

```julia
using CUDA, Spinglass

# Load and transfer to GPU
prob = load_spinglass("data/example.txt")
prob_gpu = CUDA.cu(SpinGlassSA(prob))
temps_gpu = CUDA.cu(collect(10.0:-0.1:0.1))

# Run 256 parallel annealing processes on GPU
opt_cost, opt_config = anneal(256, prob_gpu, temps_gpu, 5000)
```

### Quantum-Inspired Dynamics

```julia
using Spinglass

# Generate Sherrington-Kirkpatrick model
J = sk_model(100, 42)  # 100 spins, seed 42

# Create transverse field model
beta = zeros(1000)  # Dummy temperature array
model = Transverse(J, beta, 10;  # 10 parallel trials
    gama=2.0,    # Transverse field strength
    g=1.0,       # Damping
    a_set=-0.5   # Final field parameter
)

# Run dynamics
energy, trajectory = iterate_T(model)

# Find best result
best_trial = argmin(energy[:, end])
println("Best energy: ", energy[best_trial, end])
```

### Maximum Independent Set

```julia
using Graphs, Spinglass

# Create a spatial graph
grid = trues(10, 10)
g = unit_disk_grid_graph(grid, √2 + 1e-5)  # 8-connected lattice

# Solve MIS problem
sa = SimulatedAnnealingMIS(g)
target = -50.0  # Target set size (negative for minimization)
samples = track_equilibration!(sa, target, 100)

# Extract solution
best_set = findall(sa.IS_bitarr)
println("Independent set size: ", length(best_set))
println("Vertices: ", best_set)
```

## Examples

Run the comprehensive examples:

```bash
# Main examples (problem reduction and simulated annealing)
dir=Spinglass make example

# GPU acceleration example
julia --project=Spinglass/examples Spinglass/examples/cuda.jl

# Maximum independent set example  
julia --project=Spinglass/examples Spinglass/examples/mis.jl

# Transverse field dynamics
julia --project=Spinglass/examples Spinglass/examples/twospin.jl
```

## Problem Reductions

The package includes examples of reducing other NP-hard problems to spin glass:

1. **Circuit Satisfiability → Spin Glass**: Logic gates implemented as spin gadgets
2. **Factoring → Spin Glass**: Integer factorization via circuit compilation
3. **Maximum Independent Set → Spin Glass**: Graph problems as energy minimization

See `examples/main.jl` for demonstrations.

## API Documentation

### Core Types

- **`SpinGlassSA`**: Dense matrix representation for simulated annealing
- **`SpinConfig`**: Configuration state with cached fields
- **`Transverse`**: Quantum-inspired oscillator dynamics solver
- **`SimulatedAnnealingMIS`**: Maximum independent set solver

### Main Functions

- **`anneal`**: Simulated annealing optimization
- **`load_spinglass`**: Load problems from file
- **`random_config`**: Generate random initial states
- **`iterate_T`**: Run transverse field dynamics
- **`sk_model`**: Generate Sherrington-Kirkpatrick models
- **`unit_disk_graph`**: Create geometric graphs
- **`track_equilibration!`**: MIS with sampling

For detailed documentation, use Julia's help system:
```julia
using Spinglass
?anneal
?SpinGlassSA
?Transverse
```

## Performance Tips

### Simulated Annealing
- Use more runs (`nrun`) for better solution quality
- Slower cooling (more temperature steps) improves results
- GPU acceleration beneficial for nrun ≥ 100 and N ≥ 100

### Temperature Schedules
- **Fast**: Linear cooling, T₀:-ΔT:T_final
- **Standard**: Geometric, T₀ · α^k  
- **Thorough**: Logarithmic, T₀/log(1+k)

### Memory Efficiency
- `SpinGlassSA` uses dense matrices (O(N²) memory)
- For sparse problems, consider using `ProblemReductions.SpinGlass` directly
- GPU requires enough VRAM for nrun × N configurations

## Applications

- **Quantum Computing**: Benchmarking quantum annealers
- **Optimization**: Portfolio optimization, scheduling, routing
- **Physics**: Frustrated magnets, spin glasses, Ising models
- **Machine Learning**: Boltzmann machines, restricted Boltzmann machines
- **Rydberg Atoms**: Neutral atom quantum computing, MIS on unit disk graphs

## References

### Scientific Background
- [^Santoro2002] Santoro et al., "Theory of Quantum Annealing", Science 295, 2427 (2002)
- [^SK1975] Sherrington & Kirkpatrick, "Solvable Model of a Spin-Glass", Phys. Rev. Lett. 35, 1792 (1975)
- [^Cain2023] Cain et al., "Quantum speedup for combinatorial optimization with flat energy landscapes", arXiv:2306.13123 (2023)

### Problem Reductions
- [^Nguyen2023] Nguyen et al., "Quantum optimization with arbitrary connectivity using Rydberg atom arrays", PRX Quantum 4, 010316 (2023)
- [^Glover2019] Glover et al., "Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models", 4OR 17, 335-371 (2019)

### Educational Resources
- [^SSSS] Deep Learning and Quantum Programming: A Spring School, https://github.com/QuantumBFS/SSSS

## Contributing

This is an educational package demonstrating scientific computing techniques. Contributions, bug reports, and suggestions are welcome!

## License

See the repository LICENSE file for details.

## Citation

If you use this package in academic work, please cite:
```bibtex
@misc{spinglass2024,
  author = {Liu, Jin-Guo and contributors},
  title = {Spinglass: Scientific Computing Demos for Spin Glass Optimization},
  year = {2024},
  url = {https://github.com/GiggleLiu/ScientificComputingDemos/tree/main/Spinglass}
}
```
