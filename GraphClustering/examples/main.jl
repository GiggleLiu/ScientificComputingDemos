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
gcat = glue_graphs(g1, g2)
@info "glued graph: $gcat"

# visualization of a graph
graphviz = GraphViz(gcat)
@info "the spring layout of the glued graph: $(graphviz.locs)"
filename = "spring-layout.png"
drawing(graphviz; filename)
@info "saved the spring layout of the glued graph to `$filename`"

k = 2
sigma = 2.0
@info "spectral clustering with $k clusters, sigma=$sigma"
# Reference:
# Ng, Andrew, Michael Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems 14 (2001).
# https://papers.nips.cc/paper_files/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html
res = spectral_clustering(graphviz.locs, k; sigma)
@info "the clustering result: $(res.assignments), centers: $(res.centers)"

setcolor!(graphviz, 1:nv(gcat), ("red", "blue")[res.assignments])
filename = "two-clusters.png"
drawing(graphviz; filename)
@info "saved the clustering result to `$filename`"