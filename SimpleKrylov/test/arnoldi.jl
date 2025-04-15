using LinearAlgebra, SimpleKrylov, Test

@testset "arnoldi" begin
    # Create a sparse random matrix
    n = 100
    A = rand(n, n)
    
    # Create a random starting vector and normalize it
    q1 = randn(n)
    q1 = normalize(q1)
    
    # Run our Arnoldi iteration implementation
    H, Q = arnoldi_iteration(A, q1; maxiter=20)
    
    # Test that Q is orthonormal
    @test Q'Q ≈ I(size(Q, 2)) atol=1e-10
    
    # Test eigenvalue approximations
    # The Ritz values (eigenvalues of H) should approximate some eigenvalues of A
    evals_arnoldi = eigen(H).values
    evals_exact = eigen(A).values
    
    # Sort eigenvalues by magnitude for comparison
    sort_by_magnitude(x) = sort(x, by=abs, rev=true)
    evals_arnoldi_sorted = sort_by_magnitude(evals_arnoldi)
    evals_exact_sorted = sort_by_magnitude(evals_exact)
    
    # The largest eigenvalues should be well-approximated
    @test abs(evals_arnoldi_sorted[1]) ≈ abs(evals_exact_sorted[1]) rtol=1e-2
    
    # Test with a symmetric matrix where Arnoldi should be equivalent to Lanczos
    A_sym = A + A'
    H_sym, Q_sym = arnoldi_iteration(A_sym, q1; maxiter=20)
    
    # For symmetric matrices, H should be nearly tridiagonal
    for i in 1:size(H_sym, 1)
        for j in 1:size(H_sym, 2)
            if j > i+1
                @test abs(H_sym[i, j]) < 1e-10
            end
        end
    end
end
