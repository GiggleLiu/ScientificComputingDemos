"""
    arnoldi_iteration(A::AbstractMatrix, x0::AbstractVector; maxiter) -> (Matrix, Matrix)

Arnoldi iteration for constructing an orthonormal basis of the Krylov subspace.

The Arnoldi iteration builds an orthonormal basis for the Krylov subspace
K_m(A, x₀) = span{x₀, Ax₀, A²x₀, ..., A^(m-1)x₀} and produces an upper Hessenberg
matrix H that represents the projection of A onto this subspace.

# Arguments
- `A::AbstractMatrix`: Square matrix (can be non-symmetric)
- `x0::AbstractVector`: Initial vector
- `maxiter`: Maximum number of iterations (size of Krylov subspace)

# Returns
- `H::Matrix`: Upper Hessenberg matrix of size (m+1)×m or m×m
- `Q::Matrix`: Orthonormal basis vectors as columns (Q'*Q = I)

# Algorithm
The algorithm maintains the relation: A*Q = Q*H (if no breakdown occurs)
Uses Gram-Schmidt orthogonalization to maintain orthonormality of basis vectors.

# Properties
- For symmetric matrices, H becomes tridiagonal (equivalent to Lanczos)
- Eigenvalues of H (Ritz values) approximate eigenvalues of A
- Particularly effective for approximating extremal eigenvalues

# Example
```julia
A = rand(100, 100)
x0 = randn(100)
H, Q = arnoldi_iteration(A, x0; maxiter=30)

# Ritz values approximate eigenvalues of A
ritz_values = eigen(H).values

# Can use to solve linear systems: A*x ≈ b
# Minimize ‖b - A*x‖ in Krylov subspace
```

# See Also
- `lanczos_reorthogonalize`: Specialized version for symmetric matrices
"""
function arnoldi_iteration(A::AbstractMatrix{T}, x0::AbstractVector{T}; maxiter) where T
    # Storage for Hessenberg matrix entries (column by column)
    h = Vector{T}[]
    # Storage for orthonormal basis vectors of the Krylov subspace
    q = [normalize(x0)]
    n = length(x0)
    # Ensure A is a square matrix of appropriate dimensions
    @assert size(A) == (n, n)
    
    # Main Arnoldi iteration loop
    for k = 1:min(maxiter, n)
        # Apply the matrix to the latest basis vector
        u = A * q[k]    # generate next vector
        
        # Initialize the k-th column of the Hessenberg matrix
        hk = zeros(T, k+1)
        
        # Orthogonalize against all previous basis vectors (Gram-Schmidt process)
        for j = 1:k # subtract from new vector its components in all preceding vectors
            hk[j] = q[j]' * u  # Calculate projection coefficient
            u = u - hk[j] * q[j]  # Subtract projection
        end
        
        # Calculate the norm of the remaining vector
        hkk = norm(u)
        hk[k+1] = hkk  # This will be the subdiagonal entry
        push!(h, hk)  # Store this column of coefficients
        
        # Check for convergence or breakdown
        if abs(hkk) < 1e-8 || k >= n # stop if matrix is reducible
            break
        else
            # Normalize the new basis vector and add to our collection
            push!(q, u ./ hkk)
        end
    end

    # Construct the Hessenberg matrix H from the stored coefficients
    kmax = length(h)
    H = zeros(T, kmax, kmax)
    for k = 1:length(h)
        if k == kmax
            # Last column might be shorter if we had early termination
            H[1:k, k] .= h[k][1:k]
        else
            # Standard case: copy the full column including subdiagonal entry
            H[1:k+1, k] .= h[k]
        end
    end
    
    # Return the Hessenberg matrix and the orthonormal basis matrix
    return H, hcat(q...)
end
