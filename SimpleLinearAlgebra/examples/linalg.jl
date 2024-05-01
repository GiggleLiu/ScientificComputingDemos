# modified gram schmidt is more stable than classical gram schmidt
n = 100
A = randn(n, n)
Q1, R1 = classical_gram_schmidt(A)
Q2, R2 = modified_gram_schmidt!(copy(A))
@info norm(Q1' * Q1 - I)
@info norm(Q2' * Q2 - I)