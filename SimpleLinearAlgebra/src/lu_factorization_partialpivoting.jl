function lufact_pivot!(a::AbstractMatrix)
    n = size(a, 1)
    @assert size(a, 2) == n "size mismatch"
    m = zero(a)
    P = collect(1:n)
    # loop over columns
    @inbounds for k=1:n-1
        # search for pivot in current column
        val, p = findmax(x->abs(a[x, k]), k:n)
        p += k-1
        # find index p such that |a_{pk}| ≥ |a_{ik}| for k ≤ i ≤ n
        if p != k
            # swap rows k and p of matrix A
            for col = 1:n
                a[k, col], a[p, col] = a[p, col], a[k, col]
            end
            # swap rows k and p of matrix M
            for col = 1:k-1
                m[k, col], m[p, col] = m[p, col], m[k, col]
            end
            P[k], P[p] = P[p], P[k]
        end
        if iszero(a[k, k])
            # skip current column if it's already zero
            continue
        end
        # compute multipliers for current column
        m[k, k] = 1
        for i=k+1:n
            m[i, k] = a[i, k] / a[k, k]
        end
        # apply transformation to remaining sub-matrix
        for j=k+1:n
            akj = a[k, j]
            for i=k+1:n
                a[i,j] -= m[i,k] * akj
            end
        end
    end
    m[n, n] = 1
    return m, triu!(a), P
end


