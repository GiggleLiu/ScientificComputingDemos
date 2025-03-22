struct TimeDependent{T, FT}
    value::Vector{T}
    f!::FT
end
function value!(td::TimeDependent, t)
    td.f!(td.value, t)
    return td.value
end
value!(v::Vector, t) = v
Base.length(td::TimeDependent) = length(td.value)

struct ClassicalSpinSystem{T, CT<:Union{Vector{SVector{3, T}}, TimeDependent{SVector{3, T}}}, FT<:Union{Vector{SVector{3, T}}, TimeDependent{SVector{3, T}}}}
    topology::SimpleGraph{Int}
    coupling::CT
    bias::FT
    damping::T
    function ClassicalSpinSystem(topology::SimpleGraph{Int}, coupling::CT; bias::FT=zeros(SVector{3, T}, nv(topology)), damping::T=0.0) where {T, CT<:Union{Vector{SVector{3, T}}, TimeDependent{SVector{3, T}}}, FT<:Union{Vector{SVector{3, T}}, TimeDependent{SVector{3, T}}}}
        @assert length(coupling) == ne(topology) "Coupling must be a vector of length $(ne(topology)), got $(length(coupling))"
        @assert length(bias) == nv(topology) "bias must be a vector of length $(nv(topology)), got $(length(bias))"
        return new{T, CT, FT}(topology, coupling, bias, damping)
    end
end
random_spins(n::Int; xbias=0.0, ybias=0.0, zbias=0.0) = [normalize(SVector((randn() + xbias, randn() + ybias, randn() + zbias))) for _ in 1:n]
function energy(sys::ClassicalSpinSystem{T}, spins::Vector{SVector{3, T}}) where T
    return sum(spins[i] ⋅ sys.bias[i] for i in 1:nv(sys.topology)) + sum(sum(sys.coupling[i] .* spins[src(e)] .* spins[dst(e)]) for (i, e) in enumerate(edges(sys.topology)))
end

struct Checkpoint{T}
    spins::Vector{SVector{3, T}}
    time::T
end
function simulate!(spins::Vector{SVector{D, T}}, sys::ClassicalSpinSystem{T}; algorithm, nsteps::Int, dt::T, checkpoint_steps::Int=typemax(Int)) where {D, T}
    checkpoints = Vector{Checkpoint{T}}()
    h = field(instantiate(sys, 0.0), spins)
    for i in 1:nsteps
        # evolve the state and update the field
        sysi = instantiate(sys, (i-1)*dt)
        evolve!(spins, sysi, h, algorithm, dt)
        if mod(i, checkpoint_steps) == 0
            push!(checkpoints, Checkpoint(copy(spins), i*dt))
        end
    end
    return spins, checkpoints
end
instantiate(sys::ClassicalSpinSystem{T}, t::T) where T = ClassicalSpinSystem(sys.topology, value!(sys.coupling, t); bias=value!(sys.bias, t), damping=sys.damping)

abstract type SpinDynamicsAlgorithm end

# K-th order TrotterSuzuki algorithm
struct TrotterSuzuki{K} <: SpinDynamicsAlgorithm
    partitions::Vector{Vector{Int}}
end

function TrotterSuzuki{K}(topology::SimpleGraph) where K
    partitions = partite_vertices(topology)
    return TrotterSuzuki{K}(partitions)
end

# The TrotterSuzuki algorithm
function evolve!(spins::Vector{SVector{D, T}}, sys::ClassicalSpinSystem{T}, h::Vector{SVector{D, T}}, algorithm::TrotterSuzuki{K}, dt::T) where {K, D, T}
    @assert K == 2 "Only second order TrotterSuzuki is implemented"
    for partition in algorithm.partitions
        field!(h, sys, spins)  # update the field
        for i in partition
            # spins[i] += single_spin_dynamics(h[i], spins[i]) * dt
            # TODO: fix exp, this is not accurate since this is not a linear equation when damping is present.
            # TODO: support higher order TrotterSuzuki method
            Heff = single_spin_dynamics_operator(h[i])
            if !iszero(sys.damping)
                field_damping = sys.damping * Heff * spins[i]
                Heff += single_spin_dynamics_operator(field_damping)
            end
            spins[i] = exp(Heff * dt) * spins[i]
        end
    end
end

function single_spin_dynamics(field::SVector{3, T}, spin::SVector{3, T}) where T
    return SVector((
        -field[3] * spin[2] + field[2] * spin[3],
        field[3] * spin[1] - field[1] * spin[3],
        -field[2] * spin[1] + field[1] * spin[2]
    ))
end
function single_spin_dynamics_operator(field::SVector{3, T}) where T
    return SMatrix{3, 3, T}(
        zero(T), field[3], -field[2], -field[3], zero(T), field[1], field[2], -field[1], zero(T)
    )
end

function field(sys::ClassicalSpinSystem{T}, spins::Vector{SVector{3, T}}) where T
    h = Vector{SVector{3, T}}(undef, nv(sys.topology))
    return field!(h, sys, spins)
end
function field!(f::Vector{SVector{3, T}}, sys::ClassicalSpinSystem{T}, spins::Vector{SVector{3, T}}) where T
    @assert length(spins) == length(f) "Input must be a vector of length $(length(f)), got $(length(spins))"
    f .= sys.bias
    @inbounds for (e, J) in zip(edges(sys.topology), sys.coupling)
        i, j = src(e), dst(e)
        f[i] += J .* spins[j]
        f[j] += J .* spins[i]
    end
    return f
end


function greedy_coloring(g::SimpleGraph)
    coloring = zeros(Int, nv(g))
    for node in vertices(g)
        used_neighbour_colors = unique!([coloring[nbr] for nbr in neighbors(g, node) if coloring[nbr] != 0])
        coloring[node] = findfirst(∉(used_neighbour_colors), 1:nv(g))
    end
    return coloring
end

function dual_graph(g::SimpleGraph)
    gdual = SimpleGraph(ne(g))
    for (i, e1) in enumerate(edges(g)), (j, e2) in enumerate(edges(g))
        if src(e1) == dst(e2) || src(e1) == src(e2) || dst(e1) == dst(e2) || dst(e1) == src(e2)
            add_edge!(gdual, i, j)
        end
    end
    return gdual
end

function is_valid_coloring(g::SimpleGraph, coloring::Vector{Int})
    !any(e -> coloring[src(e)] == coloring[dst(e)], edges(g))
end

function partite_vertices(g::SimpleGraph)
    coloring = greedy_coloring(g)
    return [findall(==(i), coloring) for i in 1:maximum(coloring)]
end
partite_edges(g::SimpleGraph) = partite_vertices(dual_graph(g))
