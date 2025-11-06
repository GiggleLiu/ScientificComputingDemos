# The code is translated from the Python code written by Madelyn Cain.

"""
    unit_disk_grid_graph(grid::AbstractMatrix{Bool}, radius::Real=√2+1e-5) -> SimpleGraph

Create a unit disk graph from a masked square lattice grid.

# Arguments
- `grid::AbstractMatrix{Bool}`: Boolean mask where `true` indicates an active site
- `radius::Real`: Connection radius (default: √2 + ε for diagonal connectivity)

# Returns
- `SimpleGraph`: Graph where vertices are grid sites and edges connect nearby sites

# Description
Constructs a graph from a 2D grid where:
- Each `true` cell in the grid becomes a vertex
- Two vertices are connected if their Euclidean distance is less than `radius`

Common radius values:
- `1.0`: Only horizontal/vertical neighbors (4-connectivity)
- `√2 + ε`: Horizontal, vertical, and diagonal neighbors (8-connectivity)
- `2.0`: Includes next-nearest neighbors

# Applications
This function is useful for creating spatial interaction graphs for:
- Rydberg atom arrays (atoms on a grid with blockade interactions)
- Maximum independent set problems on lattices
- Spatial optimization problems

# Examples
```julia
# Create a 5×5 grid with a hole in the middle
grid = trues(5, 5)
grid[3, 3] = false

# 8-connected lattice graph
g = unit_disk_grid_graph(grid, √2 + 1e-5)
println("Vertices: ", nv(g), ", Edges: ", ne(g))

# 4-connected lattice graph
g4 = unit_disk_grid_graph(grid, 1.0)
println("Vertices: ", nv(g4), ", Edges: ", ne(g4))
```

# Note
You can type `√2` in Julia by entering `\\sqrt<Tab>2`, or use `sqrt(2)`.

See also: `unit_disk_graph`, `SimulatedAnnealingMIS`
"""
function unit_disk_grid_graph(grid::AbstractMatrix{Bool}, radius::Real=√2+1e-5)
    # Extract locations of active sites
    locs = [(i, j) for i=1:size(grid, 1), j=1:size(grid, 2) if grid[i,j]]
    unit_disk_graph(locs, radius)
end

"""
    unit_disk_graph(locs::AbstractVector, radius::Real) -> SimpleGraph

Create a unit disk graph from a set of point locations.

# Arguments
- `locs::AbstractVector`: Vector of locations (each element should support arithmetic)
- `radius::Real`: Connection radius

# Returns
- `SimpleGraph`: Graph where edges connect points within distance `radius`

# Description
Constructs a geometric graph where:
- Each location becomes a vertex (numbered 1 to length(locs))
- Vertices i and j are connected if ||loc[i] - loc[j]|| < radius

The distance metric is Euclidean: d(p,q) = √(Σᵢ(pᵢ-qᵢ)²)

# Complexity
O(n²) where n is the number of locations (all pairwise distances computed)

# Examples
```julia
# Random points in 2D
using Random
Random.seed!(42)
locs = [(rand(), rand()) for _ in 1:50]

# Connect points within distance 0.2
g = unit_disk_graph(locs, 0.2)
println("Created graph with ", nv(g), " vertices and ", ne(g), " edges")

# Works in 3D too
locs3d = [(rand(), rand(), rand()) for _ in 1:30]
g3d = unit_disk_graph(locs3d, 0.3)
```

# Applications
- Wireless sensor networks (communication range)
- Rydberg atom arrays (blockade radius)
- Spatial point processes
- Geometric clustering

See also: `unit_disk_grid_graph`, `SimulatedAnnealingMIS`
"""
function unit_disk_graph(locs::AbstractVector, radius::Real)
    n = length(locs)
    g = SimpleGraph(n)
    
    # Add edge if points are within radius
    for i=1:n, j=i+1:n
        if sum(abs2, locs[i] .- locs[j]) < radius ^ 2
            add_edge!(g, i, j)
        end
    end
    return g
end

"""
    SimulatedAnnealingMIS

Simulated annealing solver for the Maximum Independent Set (MIS) problem on graphs.

# Description
An independent set is a set of vertices with no edges between them. The MIS problem
seeks the largest such set. This solver uses simulated annealing with specialized
move proposals (addition, removal, spin-exchange) to explore the solution space.

The objective function being minimized is:

    E = a·|S| + b·(number of edge violations)

where |S| is the size of the independent set and edge violations count pairs of
adjacent vertices both in S. With a = -1 and b = ∞, this finds maximum independent sets.

# Fields
## Problem Definition
- `graph::SimpleGraph{Int}`: The input graph
- `a::Float64`: Single-site energy (chemical potential), typically -1.0
- `b::Float64`: Blockade penalty for adjacent vertices, typically Inf

## State Variables
- `IS_bitarr::Vector{Bool}`: Current independent set (true = vertex in set)
- `obj::Float64`: Current objective function value
- `best_obj::Float64`: Best objective value seen so far

# Constructor
    SimulatedAnnealingMIS(graph::SimpleGraph; a::Float64 = -1.0, b::Float64 = Inf)

Create an MIS solver for the given graph.

# Parameters
- `a = -1.0`: Reward for including a vertex (negative to maximize set size)
- `b = Inf`: Penalty for violations (Inf enforces independence constraint strictly)

Setting `b = Inf` means violations are never accepted, ensuring a valid independent set.
Finite `b` allows exploration of invalid configurations during annealing.

# Related Problems
- Vertex Cover: Complement of MIS
- Maximum Clique: MIS on the complement graph  
- Rydberg Atom Array Optimization: MIS with spatial constraints

# Examples
```julia
using Graphs

# Create a graph
g = grid([5, 5])  # 5×5 grid graph

# Create MIS solver
sa = SimulatedAnnealingMIS(g)

# Solve via simulated annealing
states = track_equilibration!(sa, -12.0, 100, 1e-8)

# Extract solution
best_set = findall(sa.IS_bitarr)
println("Independent set size: ", length(best_set))
println("Vertices in set: ", best_set)
```

# Algorithm Details
The solver uses three types of moves:
1. **Addition**: Try to add a vertex to the set
2. **Removal**: Try to remove a vertex from the set  
3. **Spin-exchange**: Swap a vertex in the set with a neighboring vertex out

The spin-exchange moves help escape local minima by making non-local updates.

# References
- Cain et al., Phys. Rev. Research 4, 033019 (2022)
- Garey & Johnson, "Computers and Intractability" (MIS is NP-complete)

See also: `step!`, `track_equilibration!`, `unit_disk_graph`
"""
mutable struct SimulatedAnnealingMIS
    const graph::SimpleGraph{Int}
    const a::Float64   # chemical potential (i.e. single-site energy)
    const b::Float64   # blockade energy

    # Internal states of the annealing process
    IS_bitarr::Vector{Bool}
    obj::Float64       # Current objective function value
    best_obj::Float64  # Best objective seen

    function SimulatedAnnealingMIS(graph::SimpleGraph; a::Float64 = -1.0, b::Float64 = Inf)
        # Initialize with empty independent set
        IS_bitarr = zeros(Bool, nv(graph))
        new(graph, a, b, IS_bitarr, 0.0, 0.0)
    end
end

"""
    step!(sa::SimulatedAnnealingMIS, T::Float64, node::Int)

Perform one Monte Carlo update step for MIS simulated annealing.

# Arguments
- `sa::SimulatedAnnealingMIS`: The solver state (modified in-place)
- `T::Float64`: Current temperature
- `node::Int`: Index of the node to attempt updating

# Description
Proposes and accepts/rejects one of three types of moves:
1. **Add node** (with probability `p_add = 1.0`): Add `node` to the independent set
2. **Remove node** (with probability `1 - p_add`): Remove `node` from the set
3. **Spin-exchange**: Swap `node` with one of its neighbors (if conditions allow)

The acceptance is determined by the Metropolis criterion:
- Accept if ΔE < 0 (energy decreases)
- Accept with probability exp(-ΔE/T) if ΔE > 0

# Energy Changes
When adding/removing a vertex:
- Adding: ΔE = a + b·(number of neighbors in set)
- Removing: ΔE = -a - b·(number of neighbors in set)

Spin-exchange updates the blockade penalties accordingly.

# Side Effects
Modifies `sa.IS_bitarr` and `sa.obj` in-place if the move is accepted.

# Note
The `!` suffix indicates this function modifies its first argument (Julia convention).

See also: `step_equilibrate_by_exchange!`, `track_equilibration!`
"""
function step!(sa::SimulatedAnnealingMIS, T::Float64, node::Int)
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
    step_equilibrate_by_exchange!(sa::SimulatedAnnealingMIS, T::Float64, node::Int)

Perform a spin-exchange Monte Carlo move for equilibration.

# Arguments
- `sa::SimulatedAnnealingMIS`: The solver state (modified in-place)
- `T::Float64`: Current temperature  
- `node::Int`: Index of a node in the current independent set

# Description
Attempts to swap `node` (which must be IN the set) with one of its neighbors
(which must be OUT of the set). This move maintains the size of the independent
set while potentially improving its quality.

The move proposal:
1. If `node` is in the set and has neighbors outside the set
2. Randomly select a neighbor outside the set
3. Swap them (remove `node`, add `neighbor`)
4. Accept/reject via Metropolis criterion

This type of move is useful during the "equilibration" phase after reaching a
target energy, to sample different configurations with the same energy.

# When to Use
- After reaching a target independent set size
- To explore the solution manifold at fixed |S|
- For detailed balance sampling at low temperature

# Selection Probability
A neighbor is chosen with probability min(1, degree/8), which helps maintain
detailed balance. The factor 8 approximates the maximum degree in grid graphs.

# Side Effects
Modifies `sa.IS_bitarr` and `sa.obj` if the move is accepted.

See also: `step!`, `track_equilibration!`
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

"""
    track_equilibration!(sa::SimulatedAnnealingMIS, target_obj::Float64, depth::Int, T=1e-8) -> Matrix{Bool}

Run simulated annealing until reaching a target energy, then collect equilibrated samples.

# Arguments
- `sa::SimulatedAnnealingMIS`: The solver (modified in-place)
- `target_obj::Float64`: Target objective value to reach (typically -|MIS|)
- `depth::Int`: Number of equilibrated samples to collect
- `T::Float64`: Temperature for equilibration phase (default: 1e-8, nearly zero)

# Returns
- `states::Matrix{Bool}`: Matrix of size (nv(graph), depth) where each column is a sampled configuration

# Description
This function runs in two phases:

**Phase 1 - Annealing**: Uses all three move types (add, remove, exchange) to reach
the target objective value. Updates all vertices in random order each sweep.

**Phase 2 - Equilibration**: After reaching `target_obj`, switches to spin-exchange
moves only, sampling `depth` configurations. Only vertices in the independent set
are updated during this phase.

# Typical Usage
```julia
# Find MIS for a graph
g = unit_disk_grid_graph(trues(10, 10), √2 + 1e-5)
sa = SimulatedAnnealingMIS(g)

# Anneal to target size 50, collect 100 samples
target = -50.0  # negative of desired set size
samples = track_equilibration!(sa, target, 100)

# Analyze samples
@assert all(sum(samples, dims=1) .== 50)  # All have size 50
println("Collected ", size(samples, 2), " independent sets of size 50")
```

# Physical Interpretation
At low temperature (T ≈ 0), the spin-exchange moves sample the ground state
manifold uniformly, giving different MIS configurations with the same size.
This is useful for:
- Counting ground states
- Finding diverse solutions  
- Statistical mechanics studies

# Performance Notes
- Phase 1 duration depends on problem difficulty and target
- Phase 2 is deterministic in length (exactly `depth` samples)
- Lower `T` gives more uniform sampling but slower mixing

# Side Effects
Modifies `sa.IS_bitarr`, `sa.obj`, and `sa.best_obj` throughout execution.

See also: `SimulatedAnnealingMIS`, `step!`, `step_equilibrate_by_exchange!`
"""
function track_equilibration!(sa::SimulatedAnnealingMIS, target_obj::Float64, depth::Int, T=1e-8)
    update_by_exchange = false

    # Initialize objective (trying to minimize)
    sa.obj = 0.0
    sa.best_obj = sa.obj

    mynodelist = collect(vertices(sa.graph))
    d = 1
    # Storage for equilibrated samples (column-major for efficiency)
    states = zeros(Bool, (nv(sa.graph), depth))
    exit = false
    
    while !exit
        Random.shuffle!(mynodelist)
        # Perform one sweep over all nodes
        local mynewnodelist
        for v in mynodelist
            # Choose update type based on phase
            update_by_exchange ? step_equilibrate_by_exchange!(sa, T, v) : step!(sa, T, v)
            
            # Track best objective seen
            if sa.obj < sa.best_obj
                sa.best_obj = sa.obj
            end

            # Once target is reached, switch to equilibration phase
            if sa.best_obj == target_obj
                # Now only perform spin-exchange moves on vertices in the set
                update_by_exchange = true
                mynewnodelist = findall(sa.IS_bitarr)
                
                # Collect sample
                if d <= depth
                    states[:, d] .= sa.IS_bitarr
                else
                    exit = true
                    break
                end
                d += 1
            end
        end

        # In equilibration phase, only update vertices in the set
        if sa.best_obj == target_obj
            mynodelist = mynewnodelist
        end
    end
    return states
end