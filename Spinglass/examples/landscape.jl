using Spinglass

using Random
Random.seed!(2)
const tempscales = 10 .- ((1:64) .- 1) .* 0.15 |> collect
const sap = load_spinglass(pkgdir(Spinglass, "data", "example.txt"))
 
@time anneal(30, sap, tempscales, 4000)

using LuxorGraphPlot
function multipartite_layout(graph, sets; C=2.0)
    locs = Vector{Tuple{Float64, Float64}}[]
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