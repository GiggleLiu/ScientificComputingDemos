using SimpleTensorNetwork
using SimpleTensorNetwork.Graphs, SimpleTensorNetwork.OMEinsum
using LuxorGraphPlot

graphname = :petersen
graph = smallgraph(graphname)
filename = "graph-$graphname.png"
LuxorGraphPlot.show_graph(graph; filename, texts=string.(1:nv(graph)))
@info "loaded graph: $graphname, saved to `$filename`"

sg = Spinglass(graph, ones(15), zeros(10))
@info """initialized spin-glass on graph:
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

using GenericTensorNetworks
problem = GenericTensorNetworks.SpinGlass(graph; J=fill(-1, ne(graph)))
configs = solve(problem, ConfigsMin(3; bounded=false))[]

function connect_by_hamming_distance(configs)
    nc = length(configs)
    g = SimpleGraph(nc)
    for i in 1:nc-1, j in i+1:nc
        hamming_distance(configs[i], configs[j]) <= 2 && add_edge!(g, i, j)
    end
    return g
end
hamming_distance(a, b) = sum(a .!= b)

using LuxorGraphPlot
function multipartite_layout(graph, sets; C=2.0)
    locs = Vector{Tuple{Float64, Float64}}[]
    @show sets
    for (meanloc, set) in sets
        gi, = Graphs.induced_subgraph(graph, set)
        xs, ys = LuxorGraphPlot.spring_layout(gi; C)
        f = nv(gi)^1.5/1000
        push!(locs, map(xs, ys) do x, y
            (f * x + meanloc[1], f * y + meanloc[2])
        end)
    end
    locs
end

function zstack_layout(graph, sets; C=2.0, xyratio=3, deltaz=10.0)
    @show [(0, -(k-1)*deltaz)=>s for (k, s) in enumerate(sets)]
    locs = multipartite_layout(graph, [(0, (k-1)*deltaz)=>s for (k, s) in enumerate(sets)]; C)
    return vcat(map(loc->map(x->(x[1] * xyratio, x[2]), loc), locs)...)
end

cgraph = connect_by_hamming_distance(vcat(configs.coeffs[3].data, configs.coeffs[1].data))
nc1, nc2 = length(configs.coeffs[3].data), length(configs.coeffs[1].data)
locs_zstack = zstack_layout(cgraph, [1:nc1, nc1+1:nc1+nc2]; deltaz=3)
LuxorGraphPlot.show_graph(cgraph; locs=locs_zstack, vertex_color="red", vertex_size=0.1)