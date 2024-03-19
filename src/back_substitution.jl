function back_substitution!(l::AbstractMatrix, b::AbstractVector)
    n = length(b)
    @assert size(l) == (n, n) "size mismatch"
    x = zero(b)
    # loop over columns
    for j = 1:n
        # stop if matrix is singular
        if iszero(l[j, j])
            error("The lower triangular matrix is singular!")
        end
        # compute solution component
        x[j] = b[j] / l[j, j]
        for i = j+1:n
            # update right hand side
            b[i] = b[i] - l[i, j] * x[j]
        end
    end
    return x
end
