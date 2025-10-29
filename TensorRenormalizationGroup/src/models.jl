"""
    Ising()

A struct representing the 2D Ising model for tensor renormalization group calculations.
"""
struct Ising end

"""
    model_tensor(::Ising, β)

Construct the local tensor for the 2D Ising model at inverse temperature β.
The tensor has four indices (up, right, down, left) corresponding to the
four neighboring spins on a square lattice.

The Boltzmann weight for the Ising model is:
```math
W = \\exp(\\beta \\sum_{\\langle i,j \\rangle} \\sigma_i \\sigma_j)
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
    # Identity tensor for the plaquette
    a = reshape(Float64[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2, 2, 2, 2)
    
    # Bond tensor elements (avoid hvcat for Zygote compatibility)
    cβ, sβ = sqrt(cosh(β)), sqrt(sinh(β))
    factor = 1/sqrt(2)
    
    # Build q matrix using reshape instead of literals
    q = reshape([cβ+sβ, cβ-sβ, cβ-sβ, cβ+sβ], 2, 2) * factor
    
    # Contract to form the rank-4 Ising tensor
    return ein"abcd,ai,bj,ck,dl -> ijkl"(a, q, q, q, q)
end

