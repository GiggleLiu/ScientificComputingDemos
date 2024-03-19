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
        return i == j ? one(T) : zero(T)
    end
end

function qr_left_mul!(A::AbstractMatrix, givens::GivensMatrix)
    for col in 1:size(A, 2)
        vi, vj = A[givens.i, col], A[givens.j, col]
        A[givens.i, col] = vi * givens.c + vj * givens.s
        A[givens.j, col] = -vi * givens.s + vj * givens.c
    end
    return A
end

function qr_right_mul!(A::AbstractMatrix, givens::GivensMatrix)
    for row in 1:size(A, 1)
        vi, vj = A[row, givens.i], A[row, givens.j]
        A[row, givens.i] = vi * givens.c + vj * givens.s
        A[row, givens.j] = -vi * givens.s + vj * givens.c
    end
    return A
end


function givens_matrix(A, i, j)
    x, y = A[i, 1], A[j, 1]
    norm = sqrt(x^2 + y^2)
    c = x/norm
    s = y/norm
    return GivensMatrix(c, s, i, j, size(A, 1))
end

function givens_qr!(Q::AbstractMatrix, A::AbstractMatrix)
    m, n = size(A)
    if m == 1
        return Q, A
    else
        for k = m:-1:2
            g = givens_matrix(A, k-1, k)
            qr_left_mul!(A, g)
            qr_right_mul!(Q, g)
        end
        givens_qr!(view(Q, :, 2:m), view(A, 2:m, 2:n))
        return Q, A
    end
end
