#=
LU decomposition is a method for solving linear equations that involves breaking down a matrix into lower and upper triangular matrices.
The decomposition of a matrix A is represented as A = LU, where L is a lower triangular matrix and U is an upper triangular matrix.
An elementary elimination matrix is a matrix that is used in the process of Gaussian elimination to transform a system of linear equations into an equivalent system that is easier to solve. 
It is a square matrix that is obtained by performing a single elementary row operation on the identity matrix.
=#
function elementary_elimination_matrix(A::AbstractMatrix{T}, k::Int) where T
    n = size(A, 1)
    @assert size(A, 2) == n
    # create Elementary Elimination Matrices
    M = Matrix{Float64}(I, n, n)
    for i=k+1:n
        M[i, k] =  -A[i, k] ./ A[k, k]
    end
    return M
end

# A naive implementation of elimentary elimination matrix is as follows
function lufact_naive!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    @assert size(A, 2) == n
    M = Matrix{T}(I, n, n)
    for k=1:n-1
        m = elementary_elimination_matrix(A, k)
        M = M * inv(m)
        A .= m * A
    end
    return M, A
end


#= 
The above implementation has time complexity O(n^4) and since we did not use the sparsity of elimentary elimination matrix. 
A better implementation that gives O(n^3) time complexity is as follows
=#

function lufact!(a::AbstractMatrix)
    n = size(a, 1)
    @assert size(a, 2) == n "size mismatch"
    m = zero(a)
    m[1:n+1:end] .+= 1
    # loop over columns
    for k=1:n-1
        # stop if pivot is zero
        if iszero(a[k, k])
            error("Gaussian elimination fails!")
        end
        # compute multipliers for current column
        for i=k+1:n
            m[i, k] = a[i, k] / a[k, k]
        end
        # apply transformation to remaining sub-matrix
        for j=k+1:n
            for i=k+1:n
                a[i,j] -= m[i,k] * a[k, j]
            end
        end
    end
    return m, triu!(a)
end


