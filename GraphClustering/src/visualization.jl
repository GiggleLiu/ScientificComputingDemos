"""
    GraphViz{GT<:AbstractGraph}
    GraphViz(graph)

Create a `GraphViz` object from a graph for visualization.

### Interfaces
- `setcolor!`: set the color of vertices or edges
- `setlabel!`: set the label of vertices
- `setsize!`: set the size of vertices
"""
struct GraphViz{GT<:AbstractGraph}
    graph::GT
    locs::Vector{Tuple{Float64, Float64}}

    vertex_labels::Vector{String}
    vertex_colors::Vector{String}
    vertex_sizes::Vector{Float64}

    edge_colors::Vector{String}
end
function GraphViz(g::AbstractGraph)
    GraphViz(g, [50 .* (x, y) for (x, y) in zip(LuxorGraphPlot.spring_layout(g)...)],

        fill("", nv(g)),
        fill("black", nv(g)),
        fill(0.2, nv(g)),

        fill("black", ne(g)),
    )
end

function setcolor!(gv::GraphViz, vertices::AbstractVector{Int}, color)
    gv.vertex_colors[vertices] .= color
    gv
end
function setlabel!(gv::GraphViz, vertices::AbstractVector{Int}, label)
    gv.vertex_labels[vertices] .= label
    gv
end
function setsize!(gv::GraphViz, vertices::AbstractVector{Int}, size)
    gv.vertex_sizes[vertices] .= size
    gv
end
function setcolor!(gv::GraphViz, edges::AbstractVector{Tuple{Int, Int}}, color)
    @assert all(e->has_edge(gv.graph, e...), edges)
    gv.edge_colors[edge_indices(gv.graph, edges)] .= color
    gv
end
function edge_indices(g::SimpleGraph, edgs::AbstractVector{Tuple{Int, Int}})
    dict = Dict(zip(edges(g), 1:ne(g)))
    return [dict[Edge(e...)] for e in edgs]
end

"""
    drawing(gv::GraphViz; filename=nothing)

Draw the graph with the given style.

### Arguments
- `gv::GraphViz`: the graph to draw

### Keyword Arguments
- `filename::String=nothing`: the filename to save the drawing
"""
drawing(gv::GraphViz; filename=nothing) = LuxorGraphPlot.show_graph(gv.graph, gv.locs; edge_colors=gv.edge_colors, vertex_colors=gv.vertex_colors, texts=gv.vertex_labels, vertex_sizes=gv.vertex_sizes, filename)
function Base.show(io::IO, ::MIME"text/html", gv::GraphViz)
    show(io, "text/html", drawing(gv))
end
Base.display(gv::GraphViz) = display(drawing(gv))