#=
QR Factorization Methods
========================

This module implements various QR factorization algorithms for decomposing a matrix A into A = QR where:
- Q is an orthogonal matrix (Q'Q = I)
- R is an upper triangular matrix

The following algorithms are implemented:
1. Classical Gram-Schmidt
2. Modified Gram-Schmidt
3. Givens Rotations
4. Householder Reflections
=#

"""
    classical_gram_schmidt(A::AbstractMatrix{T}) where T

Compute QR factorization using the Classical Gram-Schmidt orthogonalization process.

The algorithm constructs Q column by column using the formula:
q_k = (a_k - ∑_{i=1}^{k-1} r_{ik}q_i) / r_{kk}

Returns (Q, R) where Q is orthogonal and R is upper triangular.

Note: This method is numerically less stable than Modified Gram-Schmidt.
"""
function classical_gram_schmidt(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    
    # Process first column separately
    R[1, 1] = norm(view(A, :, 1))
    Q[:, 1] .= view(A, :, 1) ./ R[1, 1]
    
    # Process remaining columns
    for k = 2:n
        Q[:, k] .= view(A, :, k)
        # Project onto span(A[:, 1:k-1])⊥
        for j = 1:k-1
            R[j, k] = dot(view(Q, :, j), view(A, :, k))
            Q[:, k] .-= view(Q, :, j) .* R[j, k]
        end
        # Normalize the k-th column
        R[k, k] = norm(view(Q, :, k))
        Q[:, k] ./= R[k, k]
    end
    
    return Q, R
end

"""
    modified_gram_schmidt(A::AbstractMatrix{T}) where T

Compute QR factorization using the Modified Gram-Schmidt orthogonalization process.

This algorithm is numerically more stable than Classical Gram-Schmidt as it
orthogonalizes against the updated vectors at each step.

Returns (Q, R) where Q is orthogonal and R is upper triangular.
"""
function modified_gram_schmidt(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    A_copy = copy(A)  # Work on a copy to preserve the input matrix
    
    for k = 1:n
        R[k, k] = norm(view(A_copy, :, k))
        Q[:, k] .= view(A_copy, :, k) ./ R[k, k]
        for j = k+1:n
            R[k, j] = dot(view(Q, :, k), view(A_copy, :, j))
            A_copy[:, j] .-= view(Q, :, k) .* R[k, j]
        end
    end
    
    return Q, R
end

# Rename the in-place version to follow Julia conventions
"""
    modified_gram_schmidt!(A::AbstractMatrix{T}) where T

In-place version of Modified Gram-Schmidt that modifies the input matrix A.
"""
function modified_gram_schmidt!(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    
    for k = 1:n
        R[k, k] = norm(view(A, :, k))
        Q[:, k] .= view(A, :, k) ./ R[k, k]
        for j = k+1:n
            R[k, j] = dot(view(Q, :, k), view(A, :, j))
            A[:, j] .-= view(Q, :, k) .* R[k, j]
        end
    end
    
    return Q, R
end

#=
Givens Rotation Implementation
==============================
A Givens rotation is a rotation in a plane spanned by two coordinate axes.
It can be used to selectively zero out specific elements of a matrix.
=#

"""
    GivensMatrix{T} <: AbstractArray{T, 2}

Represents a Givens rotation matrix that performs a plane rotation.

Fields:
- `c::T`: cosine component of the rotation
- `s::T`: sine component of the rotation
- `i::Int`: first index of the plane
- `j::Int`: second index of the plane
- `n::Int`: dimension of the matrix
"""
struct GivensMatrix{T} <: AbstractArray{T, 2}
    c::T
    s::T
    i::Int
    j::Int
    n::Int
end

Base.size(g::GivensMatrix) = (g.n, g.n)
Base.size(g::GivensMatrix, i::Int) = i == 1 || i == 2 ? g.n : 1

function Base.getindex(g::GivensMatrix{T}, i::Int, j::Int) where T
    @boundscheck i <= g.n && j <= g.n
    if i == j
        return i == g.i || i == g.j ? g.c : one(T)
    elseif i == g.i && j == g.j
        return g.s
    elseif i == g.j && j == g.i
        return -g.s
    else
        return zero(T)
    end
end

"""
    qr_left_mul!(A::AbstractMatrix, givens::GivensMatrix)

Apply a Givens rotation from the left: G*A.
Modifies A in-place.
"""
function qr_left_mul!(A::AbstractMatrix, givens::GivensMatrix)
    i, j = givens.i, givens.j
    c, s = givens.c, givens.s
    
    for col in 1:size(A, 2)
        vi, vj = A[i, col], A[j, col]
        A[i, col] = c * vi + s * vj
        A[j, col] = -s * vi + c * vj
    end
    
    return A
end

"""
    qr_right_mul!(A::AbstractMatrix, givens::GivensMatrix)

Apply a Givens rotation from the right: A*G.
Modifies A in-place.
"""
function qr_right_mul!(A::AbstractMatrix, givens::GivensMatrix)
    i, j = givens.i, givens.j
    c, s = givens.c, givens.s
    
    for row in 1:size(A, 1)
        vi, vj = A[row, i], A[row, j]
        A[row, i] = c * vi + s * vj
        A[row, j] = -s * vi + c * vj
    end
    
    return A
end

"""
    givens_matrix(A, i, j)

Construct a Givens rotation matrix that zeros out A[j, 1] when applied to A.
"""
function givens_matrix(A, i, j)
    x, y = A[i, 1], A[j, 1]
    if y == 0
        return GivensMatrix(one(eltype(A)), zero(eltype(A)), i, j, size(A, 1))
    end
    
    if x == 0
        return GivensMatrix(zero(eltype(A)), one(eltype(A)), i, j, size(A, 1))
    end
    
    norm = sqrt(x^2 + y^2)
    c = x/norm
    s = y/norm
    return GivensMatrix(c, s, i, j, size(A, 1))
end

"""
    givens_qr!(Q::AbstractMatrix, A::AbstractMatrix)

Compute QR factorization using Givens rotations.
Modifies Q and A in-place, returning Q and R (which is stored in A).
"""
function givens_qr!(Q::AbstractMatrix, A::AbstractMatrix)
    m, n = size(A)
    
    if m == 1
        return Q, A
    else
        # Zero out elements below the diagonal in the first column
        for k = m:-1:2
            g = givens_matrix(A, k-1, k)
            qr_left_mul!(A, g)
            qr_right_mul!(Q, g)
        end
        
        # Recursively process the submatrix
        givens_qr!(view(Q, :, 2:m), view(A, 2:m, 2:n))
        
        return Q, A
    end
end

"""
    givens_qr(A::AbstractMatrix{T}) where T

Compute QR factorization using Givens rotations.
Returns (Q, R) where Q is orthogonal and R is upper triangular.
"""
function givens_qr(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = Matrix{T}(I, m, m)
    R = copy(A)
    
    return givens_qr!(Q, R)
end

#=
Householder Reflection Implementation
====================================
A Householder reflection is a transformation that reflects a vector about a hyperplane.
It can be used to zero out all elements below the diagonal in a column at once.
=#

"""
    HouseholderMatrix{T} <: AbstractArray{T, 2}

Represents a Householder reflection matrix of the form I - βvv'.

Fields:
- `v::Vector{T}`: The reflection vector
- `β::T`: Scaling factor (2/‖v‖²)
"""
struct HouseholderMatrix{T} <: AbstractArray{T, 2}
    v::Vector{T}
    β::T
end

"""
    HouseholderMatrix(v::Vector{T}) where T

Construct a Householder matrix from vector v with β = 2/‖v‖².
"""
function HouseholderMatrix(v::Vector{T}) where T
    HouseholderMatrix(v, 2/norm(v)^2)
end

# Array interfaces
Base.size(A::HouseholderMatrix) = (length(A.v), length(A.v))
Base.size(A::HouseholderMatrix, i::Int) = i == 1 || i == 2 ? length(A.v) : 1

function Base.getindex(A::HouseholderMatrix, i::Int, j::Int)
    (i == j ? one(eltype(A.v)) : zero(eltype(A.v))) - A.β * A.v[i] * conj(A.v[j])
end

# Householder matrix is unitary and Hermitian
Base.inv(A::HouseholderMatrix) = A
Base.adjoint(A::HouseholderMatrix) = A

# Apply a Householder reflection from the left: A*B. Modifies B in-place.
function LinearAlgebra.mul!(Y::AbstractMatrix, A::HouseholderMatrix, B::AbstractMatrix)
    Y .= B .- (A.β .* A.v) .* (A.v' * B)
    return Y
end

# Apply a Householder reflection from the right: A*B. Modifies Y in-place.
function LinearAlgebra.mul!(Y::AbstractMatrix, A::AbstractMatrix, B::HouseholderMatrix)
    Y .= A .- (A * (B.β .* B.v)) .* B.v'
    return Y
end

function LinearAlgebra.mul!(Y::AbstractMatrix, A::HouseholderMatrix, B::HouseholderMatrix)
    Y .= A .- (A * (B.β .* B.v)) .* B.v'
    return Y
end

"""
    householder_e1(v::AbstractVector{T}) where T

Create a Householder matrix that transforms v to a multiple of e₁ (first unit vector).
"""
function householder_e1(v::AbstractVector{T}) where T
    v = copy(v)
    α = norm(v)
    
    # Handle the zero vector case
    if α == 0
        return HouseholderMatrix(v, zero(T))
    end
    
    # Handle the case where v is already a multiple of e₁
    if length(v) == 1 || all(x -> x ≈ 0, view(v, 2:length(v)))
        return HouseholderMatrix(zeros(T, length(v)), zero(T))
    end
    
    v[1] -= α * sign(v[1])
    return HouseholderMatrix(v, 2/norm(v)^2)
end

"""
    householder_qr!(Q::AbstractMatrix{T}, A::AbstractMatrix{T}) where T

Compute QR factorization using Householder reflections.
Modifies Q and A in-place, returning Q and R (which is stored in A).
"""
function householder_qr!(Q::AbstractMatrix{T}, A::AbstractMatrix{T}) where T
    m, n = size(A)
    @assert size(Q, 2) == m
    
    if m == 1
        return Q, A
    else
        # Apply Householder matrix to zero out elements below the diagonal
        H = householder_e1(view(A, :, 1))
        mul!(A, H, A)
        
        # Update Q matrix
        mul!(Q, Q, H')
        
        # Recursively process the submatrix
        householder_qr!(view(Q, :, 2:m), view(A, 2:m, 2:n))
    end
    
    return Q, A
end

"""
    householder_qr(A::AbstractMatrix{T}) where T

Compute QR factorization using Householder reflections.
Returns (Q, R) where Q is orthogonal and R is upper triangular.

This is generally the most numerically stable QR factorization method.
"""
function householder_qr(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = Matrix{T}(I, m, m)
    R = copy(A)
    
    return householder_qr!(Q, R)
end