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

function visualize_eigenvalues(lanczos_eigenvalues, exact_eigenvalues; k = 10)
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
