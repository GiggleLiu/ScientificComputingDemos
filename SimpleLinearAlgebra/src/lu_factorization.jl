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


