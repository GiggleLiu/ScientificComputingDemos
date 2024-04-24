using DelimitedFiles  # for file reading

# load graph
graphsize = 8
graph_index = 100
graph_mask = reshape(Bool.(readdlm("mis_degeneracy_L$(graphsize).dat")[graph_index+1, 4:end]), (graphsize, graphsize))
graph = unit_disk_grid_graph(graph_mask)

# load MIS graphsize
MIS_size = Int(readdlm("mis_degeneracy_L$(graphsize).dat")[graph_index+1, 1])

# run!
track_equilibration!(SimulatedAnnealingMIS(graph), -(MIS_size - 1.0), 20000)