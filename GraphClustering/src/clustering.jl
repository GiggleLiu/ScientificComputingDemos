using Graphs, KrylovKit, LuxorGraphPlot, LinearAlgebra

function connected_components(g::SimpleGraph, kmax::Int; atol=1e-8)
    # Returns the connected components of the graph.
    lap = laplacian_matrix(g)
    # hermitian matrices have real eigenvalues
    eigvals, eigvecs = eigsolve(lap, randn(nv(g)), kmax, :SR; ishermitian=true)
    # find zero eigenvalues
    idx = findall(x->abs(x) < atol, eigvals)
    # pick any of the eigenvectors, a connected cluster have the same amplitute
    v = eigvecs[first(idx)]
    # cluster the vertices by the eigenvector
    clusters = Dict{Int, Vector{Int}}()
    for (j, x) in enumerate(v)
        key = round(Int, x * 10^8)
        if haskey(clusters, key)
            push!(clusters[key], j)
        else
            clusters[key] = [j]
        end
    end
    return collect(values(clusters))
end

function glue_graphs(g1::SimpleGraph, g2::SimpleGraph)
    g = SimpleGraph(nv(g1)+nv(g2))
    for e in edges(g1)
        add_edge!(g, src(e), dst(e))
    end
    for e in edges(g2)
        add_edge!(g, src(e)+nv(g1), dst(e)+nv(g1))
    end
    return g
end

function spectral_clustering(points::AbstractVector{Vector{T}}, k; sigma) where T
    expdist(x, y) = exp(-(norm(x-y)/sigma)^2)
    adj = expdist.(reshape(points, 1, :), points)
    D = Diagonal(inv.(sqrt.(sum(adj, dims=1))))
    normalized_adj = D * adj * D
    vals, vecs = eigsolve(normalized_adj, randn(length(points)), k, :SR; ishermitian=true)
    lap = Diagonal
end