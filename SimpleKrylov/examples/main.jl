using SimpleKrylov
using Graphs
using LinearAlgebra
using Test
using CairoMakie

function run_lanczos_example(; n = 1000, degree = 3, maxiter = 200, abstol = 1e-5)
    # Create a random regular graph
    @info """Setting up graph Laplacian:
    - vertices = $n
    - degree = $degree
    """
    graph = random_regular_graph(n, degree)

    # Get the Laplacian matrix of the graph
    A = laplacian_matrix(graph)

    # Generate a random initial vector
    q1 = randn(n)
    
    # Apply our Lanczos implementation
    @info """Running Lanczos algorithm:
    - abstol = $abstol
    - maxiter = $maxiter
    """
    T, Q = lanczos_reorthogonalize(A, q1; abstol=abstol, maxiter=maxiter)
    
    # Compute eigenvalues of the resulting tridiagonal matrix
    lanczos_eigenvalues = eigen(T).values
    
    # Compare with direct eigendecomposition
    @info "Computing exact eigenvalues for comparison"
    exact_eigenvalues = eigen(Matrix(A)).values
    
    return T, Q, lanczos_eigenvalues, exact_eigenvalues
end

function run_arnoldi_example(; n = 100, maxiter = 61)
    # Create a matrix with known eigenvalues
    λ = @. 10 + (1:n)
    A = triu(rand(n, n), 1) + diagm(λ)
    
    # Generate a random right-hand side
    b = rand(n)
    b = b / norm(b)  # Normalize the vector
    
    # Run Arnoldi iteration
    @info """Running Arnoldi algorithm:
    - matrix size = $n
    - maxiter = $maxiter
    """
    H, Q = arnoldi_iteration(A, b; maxiter=maxiter)
    
    # Compute residuals for different subspace dimensions
    resid = [norm(b); zeros(maxiter-1)]
    for m in 1:maxiter-1
        s = [norm(b); zeros(m)]
        z = H[1:m+1, 1:m] \ s
        x = Q[:, 1:m] * z
        resid[m+1] = norm(b - A*x)
    end
    
    return resid
end

function visualize_arnoldi_residual(resid)
    fig = Figure(size=(1000, 600))
    ax = Axis(fig[1, 1], 
        yscale=log10,
        xlabel="Iteration",
        ylabel="Residual (log scale)",
        title="Arnoldi Method Convergence"
    )
    scatter!(ax, 1:length(resid), resid)
    lines!(ax, 1:length(resid), resid, color=:blue, linewidth=1.5)
    
    filename = joinpath(@__DIR__, "arnoldi-residuals.png")
    save(filename, fig)
    @info "Arnoldi residual plot saved to: `$filename`"
    
    return fig
end
function visualize_eigenvalues(lanczos_eigenvalues, exact_eigenvalues; k = 20)
    # Visualize the k smallest eigenvalues
    fig = Figure()
    ax = Axis(fig[1, 1], 
        xlabel = "Index", 
        ylabel = "Eigenvalue",
        title = "Comparison of Lanczos vs Exact Eigenvalues"
    )
    
    # Sort eigenvalues
    sorted_lanczos = sort(real.(lanczos_eigenvalues))
    sorted_exact = sort(real.(exact_eigenvalues))
    
    # Plot the k smallest eigenvalues
    scatter!(ax, 1:k, sorted_exact[1:k], color = :blue, markersize = 10, label = "Exact")
    scatter!(ax, 1:k, sorted_lanczos[1:k], color = :red, markersize = 6, label = "Lanczos")
    
    axislegend(ax)
    
    filename = joinpath(@__DIR__, "lanczos-eigenvalues.png")
    save(filename, fig)
    @info "Eigenvalue comparison saved to: `$filename`"
    
    return fig
end

function test_accuracy(lanczos_eigenvalues, exact_eigenvalues; k = 5, atol = 1e-5)
    # Test if the k smallest eigenvalues match
    sorted_lanczos = sort(real.(lanczos_eigenvalues))[1:k]
    sorted_exact = sort(real.(exact_eigenvalues))[1:k]
    
    @info "Testing accuracy of $k smallest eigenvalues"
    @test sorted_lanczos ≈ sorted_exact atol=atol
    
    # Print the results
    for i in 1:k
        @info "Eigenvalue $i: Lanczos = $(sorted_lanczos[i]), Exact = $(sorted_exact[i])"
    end
end

# Run the example
T, Q, lanczos_eigenvalues, exact_eigenvalues = run_lanczos_example()

# Visualize results
visualize_eigenvalues(lanczos_eigenvalues, exact_eigenvalues)

# Test accuracy
test_accuracy(lanczos_eigenvalues, exact_eigenvalues)

# Run the Arnoldi example and get residuals
residuals = run_arnoldi_example()

# Visualize the convergence of Arnoldi method
visualize_arnoldi_residual(residuals)