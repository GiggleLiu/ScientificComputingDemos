using GraphClustering

g1 = smallgraph(:petersen)
g2 = smallgraph(:tutte)
g = glue_graphs(g1, g2)
connected_components(g, 10)

function clustering(g::SimpleGraph)
end