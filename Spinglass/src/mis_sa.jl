# The code is translated from the Python code written by Madelyn Cain.
"""
    diagonal_coupled_graph(grid::AbstractMatrix{Bool}[, radius = √2 + 1e-5]) -> SimpleGraph

Create a masked diagonal coupled square lattice graph from a specified `grid`.
"""
function unit_disk_grid_graph(grid::AbstractMatrix{Bool}, radius::Real=√2+1e-5)
    # One can type `√2` with `\sqrt<Tab>2`, or just use `sqrt(2)` instead.
    locs = [(i, j) for i=1:size(grid, 1), j=1:size(grid, 2) if grid[i,j]]
    unit_disk_graph(locs, radius)
end

"""
    unit_disk_graph(locs::AbstractVector, radius::Real) -> SimpleGraph

Create a unit disk graph with locations specified by `locs` and unit distance `radius`.
"""
function unit_disk_graph(locs::AbstractVector, radius::Real)
    n = length(locs)
    g = SimpleGraph(n)
    for i=1:n, j=i+1:n
        if sum(abs2, locs[i] .- locs[j]) < radius ^ 2
            add_edge!(g, i, j)
        end
    end
    return g
end

mutable struct SimulatedAnnealingMIS
    const graph::SimpleGraph{Int}
    const a::Float64   # chemical potential (i.e. single-site energy)
    const b::Float64   # blockade energy

    # internal states of the annealing process
    IS_bitarr::Vector{Bool}
    # obj means objective
    obj::Float64
    best_obj::Float64

    function SimulatedAnnealingMIS(graph::SimpleGraph; a::Float64 = -1.0, b::Float64 = Inf)
        # internal states of the annealing process
        IS_bitarr = zeros(Bool, nv(graph))
        new(graph, a, b, IS_bitarr, 0.0, 0.0)
    end
end

"""
Add a node with probability p_add, remove a node with probability p_remove, and spin-exchange with the
remaining probability.
T = current temperature
node = node to update (or None if wants to choose a node at random)
"""
function step!(sa::SimulatedAnnealingMIS, T::Float64, node::Int)  # ! is a part of function name, for annotating functions that may change input variables.
    @assert 0 < node <= nv(sa.graph)
    new_obj = sa.obj
    p_add = 1.0
    no_move = false
    add = false
    remove = false
    @inbounds if !sa.IS_bitarr[node]  # `@inbounds` removes the boundary check
        # With probability p_add, try to add a node (even if the proposed update violates the independence constraint)
        add = rand() < p_add
        if add
            # If not in the independent set, try to add spin
            sa.IS_bitarr[node] = true
            new_obj += sa.a
            # punish from violating the independence constraint
            for q in neighbors(sa.graph, node)
                if sa.IS_bitarr[q]
                    new_obj += sa.b
                end
            end
        else
            # attempt spin exchange
            nbs = neighbors(sa.graph, node)
            if rand() * 8 < length(nbs)
                chosen_neighbor = rand(nbs)
                if sa.IS_bitarr[chosen_neighbor]
                    # otherwise, we do nothing
                    sa.IS_bitarr[node] = true
                    sa.IS_bitarr[chosen_neighbor] = false
                    # add blockade penalties for the current node
                    for u in neighbors(sa.graph, node)
                        if sa.IS_bitarr[u] == 1
                            new_obj += sa.b
                        end
                    end

                    # remove blockade penalties from the chosen neighbor
                    for u in neighbors(sa.graph, chosen_neighbor)
                        if sa.IS_bitarr[u] && u != node
                            new_obj -= sa.b
                        end
                    end
                end
            else
                no_move = true
            end
        end
    else
        # try to spin exchange
        # first remove the node
        remove = rand() > p_add
        if remove
            # if not in the independent set, try to add spin
            sa.IS_bitarr[node] = 0
            new_obj -= sa.a
            for q in neighbors(sa.graph, node)
                if sa.IS_bitarr[q]
                    new_obj -= sa.b
                end
            end
        else
            # Attempt spin exchange
            nbs = neighbors(sa.graph, node)
            # Choose a neighbor out of 8 maximum neighbors
            if rand() * 8 < length(nbs)
                chosen_neighbor = rand(nbs)
                if !sa.IS_bitarr[chosen_neighbor]
                    # otherwise, we do nothing
                    sa.IS_bitarr[node] = false
                    sa.IS_bitarr[chosen_neighbor] = true
                    # add blockade penalties from the new chosen node
                    for u in neighbors(sa.graph, chosen_neighbor)
                        if sa.IS_bitarr[u]
                            new_obj += sa.b
                        end
                    end

                    # remove blockade penalties from the previous node
                    for u in neighbors(sa.graph, node)
                        if sa.IS_bitarr[u] && u != chosen_neighbor
                            new_obj -= sa.b
                        end
                    end
                end
            else
                no_move = true
            end
        end
    end
    if !no_move
        delta = new_obj - sa.obj

        if delta < 0
            p_acc = 1.0   # must not replace 1.0 with 1, because they are different data types! Julia dislike a variable changing data types.
        else
            p_acc = min(1.0, exp(-delta / T))
        end

        if rand() < p_acc
            sa.obj = new_obj
        else
            if add
                sa.IS_bitarr[node] = false
            elseif remove
                sa.IS_bitarr[node] = true
            else
                # we have spin exchanged
                sa.IS_bitarr[node] = !sa.IS_bitarr[node]
                sa.IS_bitarr[chosen_neighbor] = !sa.IS_bitarr[chosen_neighbor]
            end
        end
    end
end

""" 
Perform a spin-exchange update.
T = current temperature
node = node to update (or None if wants to choose a node at random)
"""
function step_equilibrate_by_exchange!(sa::SimulatedAnnealingMIS, T::Float64, node::Int)
    @assert 0 < node <= nv(sa.graph)
    new_obj = sa.obj
    @inbounds if sa.IS_bitarr[node]
        # try to spin exchange
        nbs = neighbors(sa.graph, node)
        # I guess we can use the following boolean expression to fullfill the detailed ballance
        if rand() * 8 < length(nbs)
            chosen_neighbor = rand(nbs)
            if !sa.IS_bitarr[chosen_neighbor]
                # otherwise, we do nothing
                sa.IS_bitarr[node] = false
                sa.IS_bitarr[chosen_neighbor] = true
                # add blockade penalties from the new chosen node
                for u in neighbors(sa.graph, chosen_neighbor)
                    if sa.IS_bitarr[u]
                        new_obj += sa.b
                    end
                end

                # remove blockade penalties from the previous node
                for u in neighbors(sa.graph, node)
                    if sa.IS_bitarr[u] && u != chosen_neighbor
                        new_obj -= sa.b
                    end
                end
            end

            delta = new_obj - sa.obj

            if delta < 0
                p_acc = 1.0
            else
                p_acc = min(1.0, exp(-delta / T))
            end

            if rand() < p_acc
                sa.obj = new_obj
            else
                # Revert change
                sa.IS_bitarr[node] = !sa.IS_bitarr[node]
                sa.IS_bitarr[chosen_neighbor] = !sa.IS_bitarr[chosen_neighbor]
            end
        end
    end
end

function track_equilibration!(sa::SimulatedAnnealingMIS, target_obj::Float64, depth::Int, T=1e-8)
    update_by_exchange = false

    # objective function (trying to minimize)
    sa.obj = 0.0
    sa.best_obj = sa.obj

    mynodelist = collect(vertices(sa.graph))
    d = 1
    states = zeros(Bool, (nv(sa.graph), depth))  # because Julia vectors are in column major, I exchanged rows and cols
    exit = false
    while !exit
        Random.shuffle!(mynodelist)
        # Perform a sweep, trying to update each node
        local mynewnodelist
        for v in mynodelist
            update_by_exchange ? step_equilibrate_by_exchange!(sa, T, v) : step!(sa, T, v)
            if sa.obj < sa.best_obj
                sa.best_obj = sa.obj
            end

            # Once we reach the target energy, switch to only a spin-exchange update
            if sa.best_obj == target_obj
                # Now only choose to spin-exchange nodes in the independent set
                update_by_exchange = true
                mynewnodelist = findall(sa.IS_bitarr)
                if d <= depth
                    states[:, d] .= sa.IS_bitarr
                else
                    exit = true
                    break
                end
                d += 1
            end
        end

        if sa.best_obj == target_obj
            mynodelist = mynewnodelist
        end
    end
    return states
end