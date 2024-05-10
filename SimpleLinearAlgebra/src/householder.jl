# Let v be nonzero, an m-by-m matrix P of the form I - 2v*v'/v'*v is called a Householder matrix, which is both symmetric and orthogonal. 
struct HouseholderMatrix{T} <: AbstractArray{T, 2}
    v::Vector{T}
    β::T
end

function HouseholderMatrix(v::Vector{T}) where T
    HouseholderMatrix(v, 2/norm(v, 2)^2)
end

# array interfaces
Base.size(A::HouseholderMatrix) = (length(A.v), length(A.v))
Base.size(A::HouseholderMatrix, i::Int) = i == 1 || i == 2 ? length(A.v) : 1

function Base.getindex(A::HouseholderMatrix, i::Int, j::Int)
    (i == j ? 1 : 0) - A.β * A.v[i] * conj(A.v[j])
end

# Householder matrix is unitary
Base.inv(A::HouseholderMatrix) = A
# Householder matrix is Hermitian
Base.adjoint(A::HouseholderMatrix) = A

# Left and right multiplication
function left_mul!(B, A::HouseholderMatrix)
    B .-= (A.β .* A.v) * (A.v' * B)
    return B
end

function right_mul!(A, B::HouseholderMatrix)
    A .= A .- (A * (B.β .* B.v)) * B.v'
    return A
end

function householder_e1(v::AbstractVector{T}) where T
    v = copy(v)
    v[1] -= norm(v, 2)
    return HouseholderMatrix(v, 2/norm(v, 2)^2)
end


#= Let H_k be the Householder matrix that zeros out the k-th column below the diagonal. Then we have
$H_n\ldots H_2H_1A=R$
where R is an upper triangular matrix. Then we can define the Q matrix as
$Q=H_1^TH_2^T\ldots H_n^T$, which is a unitary matrix.
=#
function householder_qr!(Q::AbstractMatrix{T}, a::AbstractMatrix{T}) where T
    m, n = size(a)
    @assert size(Q, 2) == m
    if m == 1
        return Q, a
    else
        # apply householder matrix
        H = householder_e1(view(a, :, 1))
        left_mul!(a, H)
        # update Q matrix
        right_mul!(Q, H')
        # recurse
        householder_qr!(view(Q, :, 2:m), view(a, 2:m, 2:n))
    end
    return Q, a
end
