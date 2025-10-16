using LinearAlgebra, SimpleKrylov, SparseArrays, Test

@testset "lanczos" begin
    # Create a random 3-regular graph with 1000 vertices
    n = 1000
    is = rand(1:n, 3n)
    js = rand(1:n, 3n)
    vals = randn(3n)
    # create a symmetric matrix
    A = sparse(is, js, vals)
    A += A'

    # Generate a random initial vector
    q1 = randn(n)

    # Apply our Lanczos implementation
    T, Q = lanczos_reorthogonalize(A, q1; abstol=1e-5, maxiter=200)

    # Compute eigenvalues of the resulting tridiagonal matrix
    eigenvalues = eigen(T).values


    # Find the two smallest eigenvalues using KrylovKit
    # :SR means "smallest real part"
    evals, evecs = eigen(Matrix(A))
    @test evals[1:2] ≈ eigenvalues[1:2] atol=1e-5
end
