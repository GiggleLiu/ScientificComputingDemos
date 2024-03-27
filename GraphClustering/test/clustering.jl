using Test, GraphClustering

@testset "clustering" begin
    g = GraphClustering.Graphs.smallgraph(:petersen)
    g2 = GraphClustering.Graphs.smallgraph(:tutte)
    g3 = GraphClustering.glue_graphs(g, g2)
    xs, ys = GraphClustering.LuxorGraphPlot.spring_layout(g3)
    res = spectral_clustering([[x, y] for (x, y) in zip(xs, ys)], 2; sigma=2.0)
    @show res
end