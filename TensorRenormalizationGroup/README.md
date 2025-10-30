# TensorRenormalizationGroup

A Julia implementation of the Tensor Renormalization Group (TRG) algorithm for computing thermodynamic properties of 2D classical lattice models, with full automatic differentiation support via Zygote.

## Overview

The Tensor Renormalization Group (TRG) method is a powerful numerical technique for studying 2D statistical mechanics systems. This package implements the TRG algorithm with the following features:

- **Efficient Computation**: Calculate partition functions for exponentially large lattices (2^N × 2^N sites)
- **Automatic Differentiation**: Compute thermodynamic derivatives using Zygote.jl
- **Bond Dimension Control**: Systematic approximation via SVD truncation
- **Visualization Tools**: Track and visualize the renormalization process

## Installation

```julia
using Pkg
Pkg.add(path="/path/to/TensorRenormalizationGroup")
```

## Quick Start

```julia
using TensorRenormalizationGroup

# Create Ising model tensor at inverse temperature β
β = 0.44
a = model_tensor(Ising(), β)

# Run TRG algorithm
χ = 20       # Bond dimension cutoff
niter = 10   # Number of iterations
result = trg(a, χ, niter)

println("Log partition function per site: ", result.lnZ)
```

## Features

### 1. Partition Function Computation

The TRG algorithm efficiently computes the logarithm of the partition function for 2D lattice models:

```julia
# Compute for 2D Ising model
β = 0.4  # Inverse temperature
a = model_tensor(Ising(), β)
result = trg(a, χ=20, niter=10)

# For a lattice of size 2^10 × 2^10 = 1024 × 1024
system_size = 2^niter
println("System size: $(system_size) × $(system_size)")
```

### 2. Automatic Differentiation

Compute thermodynamic quantities using automatic differentiation:

```julia
using Zygote

# Internal energy: u = -∂ln(Z)/∂β
f = β -> trg(model_tensor(Ising(), β), 20, 10).lnZ
u = -Zygote.gradient(f, β)[1]

# Specific heat: C = β² ∂²ln(Z)/∂β²
d2lnZ_dβ2 = Zygote.gradient(β -> Zygote.gradient(f, β)[1], β)[1]
C = -β^2 * d2lnZ_dβ2
```

### 3. Renormalization Process Tracking

Track the evolution of bond dimensions and tensors:

```julia
result = trg(a, χ, niter)
history = result.history

for h in history
    println("Iteration $(h.iteration): bond dimension = $(h.bond_dim)")
end
```

## Mathematical Background

### The TRG Algorithm

The TRG algorithm coarse-grains a 2D tensor network by iteratively:
1. Decomposing tensors via SVD: `T ≈ U·S·V†`
2. Truncating to bond dimension χ
3. Contracting to form new coarse-grained tensors
4. Repeating for N iterations to obtain a 2^N × 2^N lattice

### Ising Model Tensor

For the 2D Ising model with Hamiltonian:
```
H = -∑_{⟨i,j⟩} σᵢσⱼ
```

The local tensor is constructed from Boltzmann weights `exp(β σᵢσⱼ)` where σ ∈ {-1, +1}.

## Examples

### Computing Phase Transition

```julia
using TensorRenormalizationGroup, CairoMakie

# Scan temperatures
β_range = range(0.2, 0.8, length=30)
χ, niter = 20, 10

# Compute specific heat
specific_heats = Float64[]
for β in β_range
    f = β -> trg(model_tensor(Ising(), β), χ, niter).lnZ
    d2lnZ_dβ2 = Zygote.gradient(β -> Zygote.gradient(f, β)[1], β)[1]
    C = -β^2 * d2lnZ_dβ2
    push!(specific_heats, C)
end

# Plot
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Temperature", ylabel="Specific Heat")
lines!(ax, 1 ./ β_range, specific_heats)
save("phase_transition.png", fig)
```

### Convergence Study

```julia
# Test convergence with different bond dimensions
χ_values = [2, 4, 8, 16, 32]
β = 0.44

results = [trg(model_tensor(Ising(), β), χ, 10).lnZ for χ in χ_values]

# Plot convergence
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Bond Dimension χ", ylabel="ln(Z/N)")
lines!(ax, χ_values, results, marker=:circle)
save("convergence.png", fig)
```

## Running the Full Example

A comprehensive example demonstrating all features is provided:

```bash
cd examples
julia --project=. main.jl
```

This generates multiple visualizations including:
- Renormalization process evolution
- Tensor element structure
- Free energy vs temperature
- Internal energy and specific heat
- Phase transition detection
- Convergence studies

## API Reference

### Core Functions

- `trg(a, χ, niter; tol=1e-16)`: Main TRG algorithm
  - `a`: Rank-4 input tensor
  - `χ`: Bond dimension cutoff
  - `niter`: Number of iterations
  - Returns: `TRGResult` with `lnZ` and `history`

- `model_tensor(::Ising, β)`: Construct Ising model tensor
  - `β`: Inverse temperature
  - Returns: 2×2×2×2 tensor

- `num_grad(f, x; δ=1e-5)`: Numerical gradient for verification
  - `f`: Scalar function
  - `x`: Point to evaluate
  - Returns: Numerical derivative

### Types

- `Ising()`: Struct representing the 2D Ising model
- `TRGResult`: Result type containing `lnZ` and `history`

## Performance

The TRG algorithm scales as O(χ^6) per iteration, where χ is the bond dimension. Typical performance:

- χ=20, niter=10: ~1 second (1024×1024 lattice)
- χ=32, niter=12: ~10 seconds (4096×4096 lattice)

Memory usage scales as O(χ^4) for storing tensors.

## Theoretical Background

The critical temperature for the 2D Ising model is:
```
Tc = 2 / ln(1 + √2) ≈ 2.269
```

The TRG algorithm captures this phase transition through a peak in the specific heat computed via automatic differentiation.

## Testing

Run the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## References

1. Levin, M., & Nave, C. P. (2007). Tensor renormalization group approach to two-dimensional classical lattice models. *Physical Review Letters*, 99(12), 120601.

2. Liao, H. J., Liu, J. G., Wang, Lei & Xiang, T. (2019). Differentiable Programming Tensor Networks. *Physical Review X*, 9, 031041.

## License

This package is part of the ScientificComputingDemos collection.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Uses [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl) for efficient tensor contractions
- Uses [Zygote.jl](https://github.com/FluxML/Zygote.jl) for automatic differentiation
- Inspired by tensorgrad by Lei Wang
