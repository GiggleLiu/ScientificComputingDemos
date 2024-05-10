using SimpleLinearAlgebra
using LinearAlgebra
# modified gram schmidt is more stable than classical gram schmidt
n = 100
A = randn(n, n)
@info "Running Gram-Schmidt orthogonalization for a random matrix of size ($n x $n)"
Q1, R1 = classical_gram_schmidt(A)
Q2, R2 = modified_gram_schmidt!(copy(A))
@info "Error in the classical Gram-Schmidt orthogonalization: $(norm(Q1' * Q1 - I))"
@info "Error in the modified Gram-Schmidt orthogonalization: $(norm(Q2' * Q2 - I))"


n = 100
A = randn(n, n)
@info "Running LU factorization for a random matrix of size ($n x $n)"
L0, U0 = lufact!(copy(A))
@info "Without pivoting, the error is: $(sum(abs, A - L0 * U0))"

L, U, P = lufact_pivot!(copy(A))
pmat = zeros(Int, n, n)
setindex!.(Ref(pmat), 1, 1:n, P)
@info "With pivoting, the error is: $(sum(abs, pmat * A - L * U))"

n = 100
A = randn(n, n)
@info "Running QR factorization for a random matrix of size ($n x $n), with Givens rotations"
R = copy(A)
Q, R = givens_qr!(Matrix{Float64}(I, n, n), R)
@info "The factorization error is: $(norm(Q * R - A)), normalization error is $(norm(Q' * Q - I))"

@info "The Householder QR factorization"
n = 100
A = randn(n, n)
R2 = copy(A)
Q2 = Matrix{Float64}(I, n, n)
householder_qr!(Q2, R2)
@info "The factorization error is: $(norm(Q2 * R2 - A)), normalization error is $(norm(Q2' * Q2 - I))"

