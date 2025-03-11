using SimpleLinearAlgebra: classical_gram_schmidt, modified_gram_schmidt, lufact!, lufact_pivot!, givens_qr!, householder_qr!
using LinearAlgebra
using Printf

function run_gram_schmidt_example(n=100)
    println("\n=== Gram-Schmidt Orthogonalization ===")
    A = randn(n, n)
    println("Testing with random matrix of size ($n × $n)")
    
    # Classical Gram-Schmidt
    Q1, R1 = classical_gram_schmidt(A)
    classical_error = norm(Q1'Q1 - I)
    
    # Modified Gram-Schmidt
    Q2, R2 = modified_gram_schmidt(A)
    modified_error = norm(Q2'Q2 - I)
    
    # Built-in QR for comparison
    Q3, R3 = qr(A)
    builtin_error = norm(Matrix(Q3)'Matrix(Q3) - I)
    
    # Print results
    @printf "Classical G-S orthogonality error: %.2e\n" classical_error
    @printf "Modified G-S orthogonality error:  %.2e\n" modified_error
    @printf "Built-in QR orthogonality error:   %.2e\n" builtin_error
    
    # Check factorization accuracy
    @printf "Classical G-S factorization error: %.2e\n" norm(Q1*R1 - A)
    @printf "Modified G-S factorization error:  %.2e\n" norm(Q2*R2 - A)
    @printf "Built-in QR factorization error:   %.2e\n" norm(Matrix(Q3)*R3 - A)
end

function run_lu_example(n=100)
    println("\n=== LU Factorization ===")
    A = randn(n, n)
    println("Testing with random matrix of size ($n × $n)")
    
    # Without pivoting
    L0, U0 = lufact!(copy(A))
    no_pivot_error = norm(A - L0*U0)
    
    # With pivoting
    L, U, P = lufact_pivot!(copy(A))
    pmat = zeros(Int, n, n)
    setindex!.(Ref(pmat), 1, 1:n, P)
    pivot_error = norm(pmat*A - L*U)
    
    # Built-in LU for comparison
    F = lu(A)
    builtin_error = norm(F.P*A - F.L*F.U)
    
    # Print results
    @printf "LU without pivoting error:  %.2e\n" no_pivot_error
    @printf "LU with pivoting error:     %.2e\n" pivot_error
    @printf "Built-in LU error:          %.2e\n" builtin_error
end

function run_qr_example(n=100)
    println("\n=== QR Factorization ===")
    A = randn(n, n)
    println("Testing with random matrix of size ($n × $n)")
    
    # Givens QR
    R_givens = copy(A)
    Q_givens, R_givens = givens_qr!(Matrix{Float64}(I, n, n), R_givens)
    givens_fact_error = norm(Q_givens*R_givens - A)
    givens_orth_error = norm(Q_givens'Q_givens - I)
    
    # Householder QR
    R_house = copy(A)
    Q_house = Matrix{Float64}(I, n, n)
    householder_qr!(Q_house, R_house)
    house_fact_error = norm(Q_house*R_house - A)
    house_orth_error = norm(Q_house'Q_house - I)
    
    # Built-in QR for comparison
    F = qr(A)
    builtin_fact_error = norm(Matrix(F.Q)*F.R - A)
    builtin_orth_error = norm(Matrix(F.Q)'Matrix(F.Q) - I)
    
    # Print results
    println("Factorization errors:")
    @printf "  Givens QR:      %.2e\n" givens_fact_error
    @printf "  Householder QR: %.2e\n" house_fact_error
    @printf "  Built-in QR:    %.2e\n" builtin_fact_error
    
    println("Orthogonality errors:")
    @printf "  Givens QR:      %.2e\n" givens_orth_error
    @printf "  Householder QR: %.2e\n" house_orth_error
    @printf "  Built-in QR:    %.2e\n" builtin_orth_error
end

function run_all_examples(n=100)
    println("\n==== SimpleLinearAlgebra Examples ====")
    println("Matrix size: $n × $n")
    
    run_gram_schmidt_example(n)
    run_lu_example(n)
    run_qr_example(n)
    
    println("\n==== Examples Complete ====")
end

# Run all examples with default size
run_all_examples()

# Uncomment to run with a different size
# run_all_examples(200)

