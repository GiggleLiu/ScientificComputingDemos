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
