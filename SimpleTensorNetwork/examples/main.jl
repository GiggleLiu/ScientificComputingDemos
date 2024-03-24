using SimpleTensorNetwork
using SimpleTensorNetwork.Graphs, SimpleTensorNetwork.OMEinsum

graphname = :petersen
graph = smallgraph(graphname)
sg = Spinglass(graph, ones(15), zeros(10))
@info """initialized spin-glass on $graphname graph:
- edges: $(collect(edges(sg.graph)))
- J: $(sg.J)
- h: $(sg.h)
"""
β = 0.1
tn = generate_tensor_network(sg, β)
@info """generated tensor network at β=$β:
- input labels: $(tn.ixs)
- output labels: $(tn.iy)
"""
complexity_before = contraction_complexity(DynamicEinCode(tn.ixs, tn.iy), Dict(i=>2 for i in vertices(graph)))
opttn = optimize_tensornetwork(tn)
complexity_after = contraction_complexity(opttn.ein, Dict(i=>2 for i in vertices(graph)))
@info """optimized tensor network, the computational complexity change:
- original: 
- optimized: 
"""
result = partition_function(sg, β)
exact_result = partition_function_exact(sg, β)
@info "partition function: $result (exact: $exact_result)"