# Ref: https://support.dwavesys.com/hc/en-us/community/posts/1500000470701-What-are-the-cost-function-for-NAND-and-NOR-gates
struct SGGadget{WT}
    sg::SpinGlass{WT}
    inputs::Vector{Int}
    outputs::Vector{Int}
end

function sg_gadget_and()
    g = SimpleGraph(3)
    add_edge!(g, 1, 2)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    sg = SpinGlass(g, [1, -2, -2], [1, 1, -2])
    SGGadget(sg, [1, 2], [3])
end

function sg_gadget_not()
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    sg = SpinGlass(g, [1], [0, 0])
    SGGadget(sg, [1], [2])
end

function sg_gadget_or()
    g = SimpleGraph(3)
    add_edge!(g, 1, 2)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    sg = SpinGlass(g, [1, -2, -2], [-1, -1, 2])
    SGGadget(sg, [1, 2], [3])
end

function sg_gadget_arraymul()
    #   s_{i+1,j-1}  p_i
    #          \     |
    #       q_j ------------ q_j
    #                |
    #   c_{i,j} ------------ c_{i-1,j}
    #                |     \
    #                p_i     s_{i,j} 
    # variables: p_i, q_j, pq, c_{i-1,j}, s_{i+1,j-1}, c_{i,j}, s_{i,j}
    # constraints: 2 * c_{i,j} + s_{i,j} = p_i q_j + c_{i-1,j} + s_{i+1,j-1}
    sg = SpinGlass(7, Vector{Int}[], Int[])
    add_sg!(sg, sg_gadget_and().sg, [1, 2, 3])
    for (clique, weight) in [[6, 7] => 2, [6, 3]=>-2, [6, 4]=>-2, [6, 5]=>-2,
                    [7, 3]=>-1, [7, 4]=>-1, [7, 5]=>-1,
                    [3, 4]=>1, [3, 5]=>1, [4, 5]=>1]
        add_clique!(sg, clique, weight)
    end
    return SGGadget(sg, [1, 2, 4, 5], [6, 7])
end

function add_sg!(sg::SpinGlass, g::SpinGlass, vmap::Vector{Int})
    @assert length(vmap) == g.n
    mapped_cliques = [map(x->vmap[x], clique) for clique in g.cliques]
    for (clique, weight) in zip(mapped_cliques, g.weights)
        add_clique!(sg, clique, weight)
    end
    return sg
end
function add_clique!(sg::SpinGlass, clique::Vector{Int}, weight)
    for (k, c) in enumerate(sg.cliques)
        if sort(c) == sort(clique)
            sg.weights[k] += weight
            return sg
        end
    end
    push!(sg.cliques, clique)
    push!(sg.weights, weight)
    return sg
end

function ground_states(sg::SpinGlass)
    return solve(GenericTensorNetwork(sg), ConfigsMin())[]
end

function truth_table(ga::SGGadget)
    res = ground_states(ga.sg)
    output = Dict{Vector{Int}, Vector{Int}}()
    for c in res.c.data
        key = c[ga.inputs]
        if haskey(output, key)
            @assert output[key] == max(output[key], c[ga.outputs])
        else
            output[key] = c[ga.outputs]
        end
    end
    return output
end