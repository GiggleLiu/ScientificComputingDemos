#=
The Gram-Schmidt orthogonalization is a method to compute the QR factorization of a matrix A by construct an orthogonal matrix 
Q and an upper triangular matrix R.
$q_k=\left(a_k-\sum_{i=1}^{k-1}r_{ik}q_i\right)/r_{kk}$
=#
function classical_gram_schmidt(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    R[1, 1] = norm(view(A, :, 1))
    Q[:, 1] .= view(A, :, 1) ./ R[1, 1]
    for k = 2:n
        Q[:, k] .= view(A, :, k)
        # project z to span(A[:, 1:k-1])⊥
        for j = 1:k-1
            R[j, k] = view(Q, :, j)' * view(A, :, k)
            Q[:, k] .-= view(Q, :, j) .* R[j, k]
        end
        # normalize the k-th column
        R[k, k] = norm(view(Q, :, k))
        Q[:, k] ./= R[k, k]
    end
    return Q, R
end

# Modified Gram-Schmidt orthogonalization
function modified_gram_schmidt!(A::AbstractMatrix{T}) where T
    m, n = size(A)
    Q = zeros(T, m, n)
    R = zeros(T, n, n)
    for k = 1:n
        R[k, k] = norm(view(A, :, k))
        Q[:, k] .= view(A, :, k) ./ R[k, k]
        for j = k+1:n
            R[k, j] = view(Q, :, k)' * view(A, :, j)
            A[:, j] .-= view(Q, :, k) .* R[k, j]
        end
    end
    return Q, R
end
