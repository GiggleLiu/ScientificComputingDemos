# Ref:
# - https://support.dwavesys.com/hc/en-us/community/posts/1500000470701-What-are-the-cost-function-for-NAND-and-NOR-gates
# - https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010316
struct SGGadget{WT}
    sg::SpinGlass{WT}
    inputs::Vector{Int}
    outputs::Vector{Int}
end
function Base.show(io::IO, ga::SGGadget)
    println(io, "SGGadget with $(ga.sg.n) variables")
    println(io, "Inputs: $(ga.inputs)")
    println(io, "Outputs: $(ga.outputs)")
    print(io, "H = ")
    for (k, c) in enumerate(ga.sg.cliques)
        w = ga.sg.weights[k]
        iszero(w) && continue
        k == 1 || print(io, w >= 0 ? " + " : " - ")
        print(io, abs(w), "*", join(["s$ci" for ci in c], ""))
    end
end
Base.show(io::IO, ::MIME"text/plain", ga::SGGadget) = show(io, ga)

function sg_gadget_and()
    g = SimpleGraph(3)
    add_edge!(g, 1, 2)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    sg = SpinGlass(g, [1, -2, -2], [1, 1, -2])
    SGGadget(sg, [1, 2], [3])
end

function sg_gadget_set0()
    g = SimpleGraph(1)
    sg = SpinGlass(g, Int[], [-1])
    SGGadget(sg, Int[], [1])
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

function compose_multiplier(m::Int, n::Int)
    component = sg_gadget_arraymul().sg
    sg = deepcopy(component)
    modules = []
    N = 0
    newindex!() = (N += 1)
    p = [newindex!() for _ = 1:m]
    q = [newindex!() for _ = 1:n]
    out = Int[]
    spre = [newindex!() for _ = 1:m]
    for s in spre push!(modules, [sg_gadget_set0().sg, [s]]) end
    for j = 1:n
        s = [newindex!() for _ = 1:m]
        cpre = newindex!()
        push!(modules, [sg_gadget_set0().sg, [cpre]])
        for i = 1:m
            c = newindex!()
            pins = [p[i], q[j], newindex!(), cpre, spre[i], c, s[i]]
            push!(modules, [component, pins])
            cpre = c
        end
        if j == n
            append!(out, s)
            push!(out, cpre)
        else
            # update spre
            push!(out, popfirst!(s))
            push!(s, cpre)
            spre = s
        end
    end
    sg = SpinGlass(N, Vector{Int}[], Int[])
    for (m, pins) in modules
        add_sg!(sg, m, pins)
    end
    return SGGadget(sg, [p..., q...], out)
end

function set_input!(ga::SGGadget, inputs::Vector{Int})
    @assert length(inputs) == length(ga.inputs)
    for (k, v) in zip(ga.inputs, inputs)
        add_clique!(ga.sg, [k], v == 1 ? 1 : -1)  # 1 for down, 0 for up
    end
    return ga
end