using GraphClustering, Graphs, LuxorGraphPlot

# ===== Basic Graph Operations =====
println("\n=== Basic Graph Operations ===")
# Create a Petersen graph - a common test case in graph theory
g = smallgraph(:petersen)
println("Number of vertices: $(nv(g))")
println("Number of edges: $(ne(g))")
println("Edges: $(collect(edges(g)))")
println("Vertices: $(collect(vertices(g)))")
println("Neighbors of vertex 1: $(neighbors(g, 1))")
println("Degree of vertex 2: $(degree(g, 2))")

# Visualize the Petersen graph
filename = joinpath(@__DIR__, "petersen.png")
show_graph(g; filename)
println("Saved visualization of Petersen graph to '$filename'")

# ===== Graph Combination =====
println("\n=== Combining Graphs ===")
# Create two different graphs
g1 = smallgraph(:petersen)  # 10 vertices
g2 = smallgraph(:tutte)     # 46 vertices
println("Graph 1: $(nv(g1)) vertices, $(ne(g1)) edges")
println("Graph 2: $(nv(g2)) vertices, $(ne(g2)) edges")

# Combine the graphs
gcat = glue_graphs(g1, g2)
add_edge!(gcat, 1, 11)
println("Combined graph: $(nv(gcat)) vertices, $(ne(gcat)) edges")

# Visualize the combined graph
locs = LuxorGraphPlot.render_locs(gcat, SpringLayout())
filename = joinpath(@__DIR__, "combined_graph.png")
show_graph(gcat, locs; filename)
println("Saved visualization of combined graph to '$filename'")

# ===== Spectral Clustering =====
println("\n=== Spectral Clustering ===")
# Spectral clustering parameters
k = 2       # Number of clusters
sigma = 20000 # Gaussian kernel parameter

# Reference:
# Ng, Andrew, Michael Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." 
# Advances in neural information processing systems 14 (2001).
# https://papers.nips.cc/paper_files/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html

# Perform spectral clustering using node positions
res = spectral_clustering(map(x->[x[1], x[2]], locs), k; sigma)
println("Clustering assignments: $(res.assignments)")
println("Cluster centers: $(res.centers)")

# Visualize the clustering result
# Assign colors based on cluster assignments
filename = joinpath(@__DIR__, "spectral_clustering_k$(k).png")
graphviz = LuxorGraphPlot.GraphViz(gcat, locs; vertex_colors=["#E41A1C", "#377EB8"][res.assignments])
show_graph(graphviz; filename)
println("Saved clustering visualization to '$filename'")

# ===== Try Different Clustering Parameters =====
println("\n=== Exploring Different Clustering Parameters ===")
# Try with more clusters
k = 3
res_k3 = spectral_clustering(locs, k; sigma)
println("With k=3, assignments: $(res_k3.assignments)")

# Visualize with 3 clusters
filename = joinpath(@__DIR__, "spectral_clustering_k$(k).png")
graphviz = LuxorGraphPlot.GraphViz(gcat, locs; vertex_colors=["#E41A1C", "#377EB8", "#4DAF4A"][res_k3.assignments])
show_graph(graphviz; filename)
println("Saved 3-cluster visualization to '$filename'")

# Try with different sigma value
k = 2
sigma = 500.0
res_sigma5 = spectral_clustering(map(x->[x[1], x[2]], locs), k; sigma)
println("With sigma=5.0, assignments: $(res_sigma5.assignments)")

# Visualize with different sigma
graphviz = LuxorGraphPlot.GraphViz(gcat, locs; vertex_colors=["#E41A1C", "#377EB8"][res_sigma5.assignments])
filename = joinpath(@__DIR__, "spectral_clustering_sigma$(sigma).png")
show_graph(graphviz; filename)
println("Saved clustering with sigma=$sigma to '$filename'")