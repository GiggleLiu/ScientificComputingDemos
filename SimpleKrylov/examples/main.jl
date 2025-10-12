using SimpleKrylov, SimpleKrylov.LinearAlgebra, SimpleKrylov.SparseArrays
using Graphs
using CairoMakie
using Printf

"""
Krylov Subspace Methods Demo
============================

This demo shows how to use Krylov subspace methods to compute eigenvalues:
1. Lanczos algorithm - for symmetric matrices (finds smallest eigenvalues efficiently)
2. Arnoldi iteration - for general matrices (approximates eigenvalues in Krylov subspace)
"""

function run_krylov_demo()
    println("🔢 Krylov Subspace Methods Demonstration")
    println("="^50)
    
    # Run Lanczos demo
    demonstrate_lanczos()
    
    # Run Arnoldi demo
    demonstrate_arnoldi()
    
    println("\n🎉 Demo complete! Check the generated plots to see the results.")
end

"""
Demonstrate Lanczos algorithm for symmetric matrices
"""
function demonstrate_lanczos()
    println("\n📊 Lanczos Algorithm (Symmetric Eigenvalue Problem):")
    println("-"^50)
    
    # Create a sparse symmetric matrix from a graph Laplacian
    n = 1000          # Number of vertices
    degree = 3        # Degree of each vertex
    
    @info "Creating $degree-regular graph with $n vertices"
    graph = random_regular_graph(n, degree)
    A = laplacian_matrix(graph)  # Graph Laplacian is symmetric
    
    # Generate random initial vector
    q1 = randn(n)
    
    # Run Lanczos algorithm
    @info "Running Lanczos algorithm (maxiter=200, abstol=1e-5)"
    T, Q = lanczos_reorthogonalize(A, q1; abstol=1e-5, maxiter=200)
    
    # Extract eigenvalues from tridiagonal matrix
    lanczos_eigenvalues = eigen(T).values
    @info "Lanczos found $(length(lanczos_eigenvalues)) eigenvalues"
    
    # Compare with exact eigenvalues (for the smallest ones)
    @info "Computing exact eigenvalues for comparison..."
    exact_eigenvalues = eigen(Matrix(A)).values
    
    # Visualize comparison
    visualize_eigenvalues(lanczos_eigenvalues, exact_eigenvalues)
    
    # Print accuracy comparison
    k = min(5, length(lanczos_eigenvalues))
    println("\nComparison of $k smallest eigenvalues:")
    println("Index  Lanczos          Exact           Error")
    println("-"^50)
    sorted_lanczos = sort(real.(lanczos_eigenvalues))
    sorted_exact = sort(real.(exact_eigenvalues))
    for i in 1:k
        error = abs(sorted_lanczos[i] - sorted_exact[i])
        @printf("  %d    %.8f    %.8f    %.2e\n", i, sorted_lanczos[i], sorted_exact[i], error)
    end
end

"""
Demonstrate Arnoldi iteration for general matrices
"""
function demonstrate_arnoldi()
    println("\n🌀 Arnoldi Iteration (General Eigenvalue Problem):")
    println("-"^50)
    
    # Create a matrix with known spectrum
    n = 100
    λ = @. 10 + (1:n)  # Known eigenvalues: 11, 12, 13, ..., 110
    A = triu(rand(n, n), 1) + diagm(λ)  # Upper triangular + diagonal
    
    @info "Matrix size: $n × $n"
    @info "Running Arnoldi iteration (maxiter=61)"
    
    # Generate random starting vector
    b = randn(n)
    b = normalize(b)
    
    # Run Arnoldi iteration
    H, Q = arnoldi_iteration(A, b; maxiter=61)
    @info "Arnoldi basis size: $(size(Q, 2))"
    
    # Compute residuals at different iteration counts
    residuals = compute_residuals(A, b, H, Q)
    
    # Visualize convergence
    visualize_arnoldi_residuals(residuals)
    
    println("\nResidual norms (every 10 iterations):")
    println("Iteration  Residual")
    println("-"^30)
    for i in [1, 10, 20, 30, 40, 50, 60]
        if i <= length(residuals)
            @printf("  %3d      %.6e\n", i, residuals[i])
        end
    end
end

"""
Compute residuals for Arnoldi iteration at different subspace dimensions
"""
function compute_residuals(A, b, H, Q)
    maxiter = size(H, 2)
    residuals = [norm(b); zeros(maxiter-1)]
    
    for m in 1:maxiter-1
        # Solve least squares problem in Krylov subspace
        s = [norm(b); zeros(m)]
        z = H[1:m+1, 1:m] \ s
        x = Q[:, 1:m] * z
        
        # Compute residual
        residuals[m+1] = norm(b - A*x)
    end
    
    return residuals
end

"""
Visualize comparison of Lanczos vs exact eigenvalues
"""
function visualize_eigenvalues(lanczos_eigenvalues, exact_eigenvalues; k=20)
    fig = Figure(size=(1000, 600))
    ax = Axis(fig[1, 1],
        xlabel="Index",
        ylabel="Eigenvalue",
        title="Lanczos vs Exact Eigenvalues ($(k) smallest)"
    )
    
    # Sort and take k smallest
    sorted_lanczos = sort(real.(lanczos_eigenvalues))[1:k]
    sorted_exact = sort(real.(exact_eigenvalues))[1:k]
    
    # Plot comparison
    scatter!(ax, 1:k, sorted_exact, color=:blue, markersize=12, label="Exact")
    scatter!(ax, 1:k, sorted_lanczos, color=:red, markersize=8, marker=:xcross, label="Lanczos")
    
    axislegend(ax, position=:rb)
    
    filename = joinpath(@__DIR__, "lanczos-eigenvalues.png")
    save(filename, fig)
    @info "Eigenvalue comparison saved to: $filename"
    
    return fig
end

"""
Visualize Arnoldi method convergence
"""
function visualize_arnoldi_residuals(residuals)
    fig = Figure(size=(1000, 600))
    ax = Axis(fig[1, 1],
        yscale=log10,
        xlabel="Iteration",
        ylabel="Residual Norm (log scale)",
        title="Arnoldi Method Convergence"
    )
    
    # Plot residuals
    iterations = 1:length(residuals)
    lines!(ax, iterations, residuals, color=:blue, linewidth=2)
    scatter!(ax, iterations, residuals, color=:blue, markersize=8)
    
    filename = joinpath(@__DIR__, "arnoldi-residuals.png")
    save(filename, fig)
    @info "Arnoldi residual plot saved to: $filename"
    
    return fig
end

# Run the demo
run_krylov_demo()
