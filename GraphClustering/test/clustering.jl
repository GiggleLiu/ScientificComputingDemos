using Test, GraphClustering, GraphClustering.Graphs

@testset "clustering" begin
    g = GraphClustering.Graphs.smallgraph(:petersen)
    g2 = GraphClustering.Graphs.smallgraph(:tutte)
    g3 = GraphClustering.glue_graphs(g, g2)
    xs = vcat([rand() for x in 1:nv(g)], [rand() + 2 for x in 1:nv(g2)])
    ys = vcat([rand() for y in 1:nv(g)], [rand() + 2 for y in 1:nv(g2)])
    res = spectral_clustering([[x, y] for (x, y) in zip(xs, ys)], 2; sigma=2.0)
    @test res.assignments == vcat(ones(Int, nv(g)), 2 .* ones(Int, nv(g2))) || res.assignments == vcat(2 .* ones(Int, nv(g)), ones(Int, nv(g2)))
end