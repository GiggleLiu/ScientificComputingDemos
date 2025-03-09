"""
    lufact(a::AbstractMatrix)

Perform LU factorization on a copy of matrix `a` without pivoting.
Returns a tuple of (L, U) matrices where A = LU.
"""
lufact(a::AbstractMatrix) = lufact!(copy(a))

#=
LU factorization (or Gaussian elimination) with row pivoting is defined as PA = LU, where:
- P is a permutation matrix
- L is a lower triangular matrix with ones on the diagonal
- U is an upper triangular matrix

This factorization is useful for solving linear systems Ax = b efficiently.
=#
"""
    lufact_pivot!(a::AbstractMatrix{T}) where T

Perform in-place LU factorization with partial pivoting (PA = LU).
Returns a tuple of (L, U, P) where:
- L is a lower triangular matrix with ones on the diagonal
- U is an upper triangular matrix
- P is a permutation vector such that A[P,:] = LU

This factorization is useful for solving linear systems Ax = b efficiently,
especially for matrices that would otherwise have small pivots.

# Arguments
- `a::AbstractMatrix{T}`: The input matrix to be factorized (will be modified)

# Returns
- `m::Matrix{T}`: The lower triangular factor L
- `a::AbstractMatrix{T}`: The upper triangular factor U (overwrites input)
- `P::Vector{Int}`: Permutation vector

# Note
The matrix must be square.
"""
function lufact_pivot!(a::AbstractMatrix{T}) where T
    n = size(a, 1)
    @assert size(a, 2) == n "Matrix must be square"
    m = zeros(T, n, n)
    P = collect(1:n)
    
    # Loop over columns
    @inbounds for k=1:n-1
        # Find pivot (largest absolute value in current column)
        pivot_val = abs(a[k,k])
        pivot_idx = k
        for i=k+1:n
            if abs(a[i,k]) > pivot_val
                pivot_val = abs(a[i,k])
                pivot_idx = i
            end
        end
        
        # Swap rows if necessary
        if pivot_idx != k
            # Swap rows k and pivot_idx of matrix A
            for col = 1:n
                a[k, col], a[pivot_idx, col] = a[pivot_idx, col], a[k, col]
            end
            # Swap rows k and pivot_idx of matrix M
            for col = 1:k-1
                m[k, col], m[pivot_idx, col] = m[pivot_idx, col], m[k, col]
            end
            P[k], P[pivot_idx] = P[pivot_idx], P[k]
        end
        
        # Skip if pivot is zero (matrix is singular)
        if iszero(a[k, k])
            continue
        end
        
        # Compute multipliers and update submatrix
        m[k, k] = one(T)
        for i=k+1:n
            m[i, k] = a[i, k] / a[k, k]
            # Apply transformation directly (more efficient)
            for j=k+1:n
                a[i,j] -= m[i,k] * a[k,j]
            end
            # Zero out elements below diagonal
            a[i,k] = zero(T)
        end
    end
    
    # Set the last diagonal element of L
    m[n, n] = one(T)
    
    return m, a, P
end

"""
    forward_substitution!(l::AbstractMatrix{T}, b::AbstractVector{T}) where T

Solve a lower triangular system Lx = b using forward substitution.

The algorithm computes each component of x sequentially:
- x₁ = b₁/L₁₁
- xᵢ = (bᵢ - ∑ⱼ₌₁ᶦ⁻¹ Lᵢⱼxⱼ)/Lᵢᵢ for i=2,...,n

# Arguments
- `l::AbstractMatrix{T}`: Lower triangular matrix
- `b::AbstractVector{T}`: Right-hand side vector

# Returns
- `x::Vector{T}`: Solution vector

# Throws
- `ErrorException`: If the matrix is singular (has zeros on diagonal)
"""
function forward_substitution!(l::AbstractMatrix{T}, b::AbstractVector{T}) where T
    n = length(b)
    @assert size(l) == (n, n) "Matrix and vector dimensions must match"
    x = similar(b)
    
    for i = 1:n
        # Check for singularity
        if iszero(l[i, i])
            error("The lower triangular matrix is singular")
        end
        
        # Compute solution component
        s = b[i]
        for j = 1:i-1
            s -= l[i, j] * x[j]
        end
        x[i] = s / l[i, i]
    end
    
    return x
end

"""
    backward_substitution!(u::AbstractMatrix{T}, b::AbstractVector{T}) where T

Solve an upper triangular system Ux = b using backward substitution.

The algorithm computes each component of x in reverse order:
- xₙ = bₙ/Uₙₙ
- xᵢ = (bᵢ - ∑ⱼ₌ᵢ₊₁ⁿ Uᵢⱼxⱼ)/Uᵢᵢ for i=n-1,...,1

# Arguments
- `u::AbstractMatrix{T}`: Upper triangular matrix
- `b::AbstractVector{T}`: Right-hand side vector

# Returns
- `x::Vector{T}`: Solution vector

# Throws
- `ErrorException`: If the matrix is singular (has zeros on diagonal)
"""
function backward_substitution!(u::AbstractMatrix{T}, b::AbstractVector{T}) where T
    n = length(b)
    @assert size(u) == (n, n) "Matrix and vector dimensions must match"
    x = similar(b)
    
    for i = n:-1:1
        # Check for singularity
        if iszero(u[i, i])
            error("The upper triangular matrix is singular")
        end
        
        # Compute solution component
        s = b[i]
        for j = i+1:n
            s -= u[i, j] * x[j]
        end
        x[i] = s / u[i, i]
    end
    
    return x
end

"""
    lufact!(a::AbstractMatrix{T}) where T

Perform in-place LU factorization without pivoting.
Decomposes matrix A into L and U where:
- L is lower triangular with ones on the diagonal
- U is upper triangular
- A = LU

This implementation has O(n³) time complexity.

# Arguments
- `a::AbstractMatrix{T}`: The input matrix to be factorized (will be modified)

# Returns
- `m::Matrix{T}`: The lower triangular factor L
- `a::AbstractMatrix{T}`: The upper triangular factor U (overwrites input)

# Throws
- `ErrorException`: If a zero pivot is encountered (use `lufact_pivot!` for such matrices)

# Note
The matrix must be square.
"""
function lufact!(a::AbstractMatrix{T}) where T
    n = size(a, 1)
    @assert size(a, 2) == n "Matrix must be square"
    
    # Initialize L matrix with ones on diagonal
    m = zeros(T, n, n)
    for i=1:n
        m[i,i] = one(T)
    end
    
    # Loop over columns
    for k=1:n-1
        # Check for zero pivot
        if iszero(a[k, k])
            error("Zero pivot encountered. Use lufact_pivot! for matrices requiring pivoting.")
        end
        
        # Compute multipliers and update submatrix
        for i=k+1:n
            m[i, k] = a[i, k] / a[k, k]
            for j=k+1:n
                a[i,j] -= m[i,k] * a[k,j]
            end
            # Zero out elements below diagonal
            a[i,k] = zero(T)
        end
    end
    
    return m, a
end

"""
    solve_lu(L::AbstractMatrix{T}, U::AbstractMatrix{T}, b::AbstractVector{T}) where T

Solve a linear system Ax = b using LU factorization without pivoting.

# Arguments
- `L::AbstractMatrix{T}`: Lower triangular matrix from LU factorization
- `U::AbstractMatrix{T}`: Upper triangular matrix from LU factorization
- `b::AbstractVector{T}`: Right-hand side vector

# Returns
- `x::Vector{T}`: Solution vector
"""
function solve_lu(L::AbstractMatrix{T}, U::AbstractMatrix{T}, b::AbstractVector{T}) where T
    # Solve Ly = b using forward substitution
    y = forward_substitution!(L, copy(b))
    
    # Solve Ux = y using backward substitution
    x = backward_substitution!(U, y)
    
    return x
end

"""
    solve_lu_pivot(L::AbstractMatrix{T}, U::AbstractMatrix{T}, P::Vector{Int}, b::AbstractVector{T}) where T

Solve a linear system Ax = b using LU factorization with pivoting.

# Arguments
- `L::AbstractMatrix{T}`: Lower triangular matrix from LU factorization
- `U::AbstractMatrix{T}`: Upper triangular matrix from LU factorization
- `P::Vector{Int}`: Permutation vector from LU factorization
- `b::AbstractVector{T}`: Right-hand side vector

# Returns
- `x::Vector{T}`: Solution vector
"""
function solve_lu_pivot(L::AbstractMatrix{T}, U::AbstractMatrix{T}, P::Vector{Int}, b::AbstractVector{T}) where T
    # Apply permutation to b
    pb = b[P]
    
    # Solve Ly = Pb using forward substitution
    y = forward_substitution!(L, pb)
    
    # Solve Ux = y using backward substitution
    x = backward_substitution!(U, y)
    
    return x
end

