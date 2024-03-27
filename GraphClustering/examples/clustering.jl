using GraphClustering

g1 = smallgraph(:petersen)
g2 = smallgraph(:tutte)
g = glue_graphs(g1, g2)