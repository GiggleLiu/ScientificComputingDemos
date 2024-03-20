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
