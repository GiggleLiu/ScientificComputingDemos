@doc raw"""
    Ising()

A struct representing the 2D Ising model for tensor renormalization group calculations.
"""
struct Ising end

@doc raw"""
    model_tensor(::Ising, β)

Construct the local tensor for the 2D Ising model at inverse temperature β.
The tensor has four indices (up, right, down, left) corresponding to the
four neighboring spins on a square lattice.

The Boltzmann weight for the Ising model is:
```math
W = \exp(\beta \sum_{\langle i,j \rangle} \sigma_i \sigma_j)
```

where σᵢ ∈ {-1, +1} are spin variables.

The local tensor is constructed as:
```
    |1
4--[a]--2
   3|
```

# Arguments
- `β::Real`: The inverse temperature (β = 1/T)

# Returns
- `Array{Float64,4}`: A 2×2×2×2 tensor representing the local Boltzmann weight

# Example
```julia
β = 0.4
a = model_tensor(Ising(), β)
χ, niter = 5, 5
lnZ = trg(a, χ, niter)
```
"""
function model_tensor(::Ising, β::Real)
    # The bond tensor Q[σᵢ, σⱼ] = exp(β σᵢ σⱼ)
    # where σ ∈ {-1, +1} mapped to indices {1, 2}
    Q = [exp(β) exp(-β);
         exp(-β) exp(β)]
    
    # Construct the rank-4 tensor by contracting four bond tensors
    # The indices form a plaquette: up-right-down-left
    # This creates the local tensor for the 2D square lattice
    return ein"ij,jk,kl,li -> ijkl"(Q, Q, Q, Q)
end

