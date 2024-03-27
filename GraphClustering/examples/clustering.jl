using GraphClustering, GraphClustering.Graphs, GraphClustering.LuxorGraphPlot

# graph operations
g = smallgraph(:petersen)
edges(g)
vertices(g)
neighbors(g, 1)
degree(g, 2)

# gluing two graphs
g1 = smallgraph(:petersen)
g2 = smallgraph(:tutte)
g = glue_graphs(g1, g2)

graphviz = GraphViz(g)
setcolor!(graphviz, 1:nv(g1), "red")
filename = "two-clusters.png"
drawing(graphviz; filename)

k = 2
locations = LuxorGraphPlot.spring_layout(g)
res = spectral_clustering([[x, y] for (x, y) in zip(locations...)], k; sigma=2.0)
gv2 = GraphViz(g)
setcolor!(gv2, 1:nv(g), ("red", "blue")[res.assignments])