"""
    lanczos(A, q1::AbstractVector; abstol, maxiter) -> (SymTridiagonal, Matrix)

Basic Lanczos algorithm for symmetric matrices without reorthogonalization.

The Lanczos algorithm builds an orthonormal basis for the Krylov subspace
K_m(A, q₁) = span{q₁, Aq₁, A²q₁, ..., A^(m-1)q₁} and produces a tridiagonal
matrix T that approximates A in this subspace.

# Arguments
- `A`: Symmetric matrix (can be sparse)
- `q1::AbstractVector`: Initial vector
- `abstol`: Absolute tolerance for convergence (based on residual norm)
- `maxiter`: Maximum number of iterations

# Returns
- `T::SymTridiagonal`: Tridiagonal matrix with eigenvalues approximating those of A
- `Q::Matrix`: Orthonormal basis vectors as columns (Q'AQ ≈ T)

# Algorithm
The algorithm maintains the relation: A*Q = Q*T + β_{k}*q_{k+1}*e_k'

# Note
This basic version may suffer from loss of orthogonality in finite precision.
Use `lanczos_reorthogonalize` for better numerical stability.

# Example
```julia
A = Symmetric(rand(100, 100))
q1 = randn(100)
T, Q = lanczos(A, q1; abstol=1e-5, maxiter=50)
eigenvalues = eigen(T).values  # Approximate eigenvalues of A
```
"""
function lanczos(A, q1::AbstractVector{T}; abstol, maxiter) where T
    # Normalize the initial vector
    q1 = normalize(q1)
    
    # Initialize storage for basis vectors and tridiagonal matrix elements
    q = [q1]                # Orthonormal basis vectors
    α = [q1' * (A * q1)]    # Diagonal elements of tridiagonal matrix
    
    # Compute first residual: r₁ = Aq₁ - α₁q₁
    Aq1 = A * q1
    rk = Aq1 .- α[1] .* q1
    β = [norm(rk)]          # Off-diagonal elements of tridiagonal matrix
    
    # Main Lanczos iteration
    for k = 2:min(length(q1), maxiter)
        # Compute next basis vector: q_k = r_{k-1}/β_{k-1}
        push!(q, rk ./ β[k-1])
        
        # Compute A*q_k
        Aqk = A * q[k]
        
        # Compute diagonal element: α_k = q_k' * A * q_k
        push!(α, q[k]' * Aqk)
        
        # Compute residual: r_k = A*q_k - α_k*q_k - β_{k-1}*q_{k-1}
        # This enforces orthogonality to the previous two vectors
        rk = Aqk .- α[k] .* q[k] .- β[k-1] * q[k-1]
        
        # Compute the norm of the residual for the off-diagonal element
        nrk = norm(rk)
        
        # Check for convergence or maximum iterations
        if abs(nrk) < abstol || k == length(q1)
            break
        end
        
        push!(β, nrk)
    end
    
    # Return the tridiagonal matrix T and orthogonal matrix Q
    return SymTridiagonal(α, β), hcat(q...)
end

"""
    lanczos_reorthogonalize(A, q1::AbstractVector; abstol, maxiter) -> (SymTridiagonal, Matrix)

Lanczos algorithm with full reorthogonalization using Householder transformations.

This numerically stable version maintains orthogonality of the Krylov basis vectors
by applying Householder transformations at each iteration. This prevents loss of
orthogonality that can occur in the basic Lanczos algorithm due to finite precision
arithmetic.

# Arguments
- `A`: Symmetric matrix (can be sparse)
- `q1::AbstractVector`: Initial vector
- `abstol`: Absolute tolerance for convergence (based on residual norm)
- `maxiter`: Maximum number of iterations

# Returns
- `T::SymTridiagonal`: Tridiagonal matrix with eigenvalues approximating those of A
- `Q::Matrix`: Orthonormal basis vectors as columns (Q'*Q = I)

# Algorithm
Uses Householder transformations to maintain orthogonality:
- At iteration k, applies all previous Householder transformations to the residual
- Constructs basis vectors that are guaranteed to be orthogonal

# Complexity
- Time: O(k²n) for k iterations on n×n matrix (more expensive than basic Lanczos)
- Space: O(kn) to store k Householder vectors

# Example
```julia
using SparseArrays
A = sprand(1000, 1000, 0.01)
A = A + A'  # Make symmetric
q1 = randn(1000)
T, Q = lanczos_reorthogonalize(A, q1; abstol=1e-5, maxiter=100)
@assert Q' * Q ≈ I  # Basis is orthonormal
eigenvalues = eigen(T).values  # Accurate eigenvalue approximations
```

# See Also
- `lanczos`: Basic version without reorthogonalization (faster but less stable)
"""
function lanczos_reorthogonalize(A, q1::AbstractVector{T}; abstol, maxiter) where T
    n = length(q1)
    
    # Normalize the initial vector
    q1 = normalize(q1)
    
    # Initialize storage
    q = [q1]                # Orthonormal basis vectors
    α = [q1' * (A * q1)]    # Diagonal elements of tridiagonal matrix
    Aq1 = A * q1
    rk = Aq1 .- α[1] .* q1
    β = [norm(rk)]          # Off-diagonal elements of tridiagonal matrix
    
    # Store Householder transformations for reorthogonalization
    householders = [householder_matrix(q1)]
    
    # Main Lanczos iteration with reorthogonalization
    for k = 2:min(n, maxiter)
        # Step 1: Apply all previous Householder transformations to residual vector
        # This ensures full orthogonality to all previous vectors
        for j = 1:k-1
            left_mul!(view(rk, j:n), householders[j])
        end
        
        # Create new Householder transformation for the current residual
        push!(householders, householder_matrix(view(rk, k:n)))
        
        # Step 2: Compute the k-th orthonormal vector by applying Householder transformations
        # Start with unit vector e_k and apply all Householder transformations in reverse
        qk = zeros(T, n)
        qk[k] = 1  # qₖ = H₁H₂…Hₖeₖ
        for j = k:-1:1
            left_mul!(view(qk, j:n), householders[j])
        end
        push!(q, qk)
        
        # Compute A*q_k
        Aqk = A * q[k]
        
        # Compute diagonal element: α_k = q_k' * A * q_k
        push!(α, q[k]' * Aqk)
        
        # Compute residual: r_k = A*q_k - α_k*q_k - β_{k-1}*q_{k-1}
        rk = Aqk .- α[k] .* q[k] .- β[k-1] * q[k-1]
        
        # Compute the norm of the residual
        nrk = norm(rk)
        
        # Check for convergence or maximum iterations
        if abs(nrk) < abstol || k == n
            break
        end
        
        push!(β, nrk)
    end
    
    # Return the tridiagonal matrix T and orthogonal matrix Q
    return SymTridiagonal(α, β), hcat(q...)
end

"""
    HouseholderMatrix{T} <: AbstractArray{T, 2}

Efficient representation of a Householder transformation matrix H = I - β*v*v'.

# Fields
- `v::Vector{T}`: Householder vector
- `β::T`: Scaling factor (typically β = 2/‖v‖²)

# Notes
Householder matrices are orthogonal reflections. Instead of storing the full
n×n matrix, we only store the vector v and scalar β, which is much more efficient.
"""
struct HouseholderMatrix{T} <: AbstractArray{T, 2}
    v::Vector{T}    # Householder vector
    β::T            # Scaling factor
end

"""
    left_mul!(B, A::HouseholderMatrix) -> B

Apply Householder transformation to matrix B in-place: B ← (I - β*v*v')*B

# Arguments
- `B`: Matrix to transform (modified in-place)
- `A::HouseholderMatrix`: Householder transformation

# Returns
- Modified matrix B

# Algorithm
Uses the Sherman-Morrison formula to apply the rank-1 update efficiently:
(I - β*v*v')*B = B - β*v*(v'*B)
This requires only O(mn) operations for an m×n matrix B.
"""
function left_mul!(B, A::HouseholderMatrix)
    # Compute v'*B
    vB = A.v' * B
    # Apply transformation: B = B - β*v*(v'*B)
    B .-= (A.β .* A.v) * vB
    return B
end

"""
    householder_matrix(v::AbstractVector) -> HouseholderMatrix

Create a Householder matrix that reflects v to a multiple of e₁.

# Arguments
- `v::AbstractVector`: Vector to reflect

# Returns
- `HouseholderMatrix`: Householder transformation H such that Hv = ±‖v‖e₁

# Algorithm
Constructs H = I - β*u*u' where:
- u = v - ‖v‖e₁  (or v + ‖v‖e₁ for numerical stability)
- β = 2/‖u‖²

# Example
```julia
v = [3.0, 4.0, 0.0]
H = householder_matrix(v)
result = H * v  # ≈ [±5.0, 0.0, 0.0]
```
"""
function householder_matrix(v::AbstractVector{T}) where T
    v = copy(v)
    # Modify first element to ensure numerical stability
    v[1] -= norm(v, 2)
    # Compute scaling factor β = 2/||v||²
    return HouseholderMatrix(v, 2/norm(v, 2)^2)
end
