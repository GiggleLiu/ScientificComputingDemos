"""
    connected_components(g::SimpleGraph, kmax::Int; atol=1e-8)

Returns the connected components of the graph using the spectral graph theory.

### Arguments
- `g::SimpleGraph`: the input graph.
- `kmax::Int`: the number of eigenvectors to compute.

### Keyword arguments
- `atol::Real`: the tolerance for the zero eigenvalues.
"""
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

"""
    glue_graphs(g1::SimpleGraph, g2::SimpleGraph)

Glue two graphs together.
"""
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

"""
    spectral_clustering(points::AbstractVector, k; sigma)

Spectral clustering algorithm.

### Arguments
- `points::AbstractVector`: the data points to be clustered.
- `k::Int`: the number of clusters.

### Keyword arguments
- `sigma::Real`: the parameter for the Gaussian kernel.

### Reference
- Ng, Andrew, Michael Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems 14 (2001).
https://papers.nips.cc/paper_files/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html
"""
function spectral_clustering(points::AbstractVector, k; sigma)
    expdist(x, y) = exp(-(sum(abs2, x .- y)/sigma)^2)
    adj = expdist.(reshape(points, 1, :), points)
    D = Diagonal(inv.(sqrt.(dropdims(sum(adj; dims=1); dims=1))))
    normalized_adj = D * adj * D
    vals, _vecs = eigsolve(-normalized_adj, randn(length(points)), k, :SR; ishermitian=true)
    vecs = hcat(_vecs[1:k]...)
    # normalize along the row
    vecs ./= sqrt.(sum(abs2, vecs; dims=2))
    return kmeans(vecs', 2)
end
onehot(k, i) = Float64[i == j for j in 1:k]