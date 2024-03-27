using Test, GraphClustering, GraphClustering.Graphs

@testset "visualization" begin
    graph = smallgraph(:petersen)
    g = GraphViz(graph)
    @test g isa GraphViz

    setcolor!(g, 1:3, "red")
    setcolor!(g, [(1,2)], "blue")
    setlabel!(g, 1:2, ["x", "y"])
    setsize!(g, 2:3, [0.3, 0.3])
    @test GraphClustering.drawing(g) isa GraphClustering.LuxorGraphPlot.Drawing
end