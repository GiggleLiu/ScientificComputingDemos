struct SpinVector{D, T}
    vec::NTuple{D, T}
end
function flatten(spins::Vector{SpinVector{D, T}}) where {D, T}
    res = Vector{T}(undef, D * length(spins))
    for j in 1:length(spins), i in 1:D
        res[i + (j-1) * D] = spins[j].vec[i]
    end
    return res
end
function unflatten!(spins::Vector{SpinVector{D, T}}, vec::Vector{T}) where {D, T}
    @assert length(vec) == D * length(spins) "Input must be a vector of length $(D * length(spins)), got $(length(vec))"
    for j in 1:length(spins)
        spins[j] = SpinVector(ntuple(i -> vec[i + (j-1) * D], D))
    end
    return spins
end

struct ClassicalSpinSystem{T}
    topology::SimpleGraph{Int}
    coupling::Vector{T}
end

struct ClassicalSpinHamiltonian{T} <: AbstractMatrix{T}
    field::Vector{T}
end
LinearAlgebra.ishermitian(h::ClassicalSpinHamiltonian) = false
srange(k::Int) = 3 * (k-1) + 1:3 * k
ClassicalSpinHamiltonian(::Type{T}; nspins::Int) where T = ClassicalSpinHamiltonian(Vector{T}(undef, nspins * 3))
Base.size(h::ClassicalSpinHamiltonian) = (length(h.field), length(h.field))
Base.size(h::ClassicalSpinHamiltonian, i::Int) = size(h)[i]
function LinearAlgebra.mul!(res::Vector{T}, h::ClassicalSpinHamiltonian{T}, v::Vector{T}) where T
    @assert length(v) == size(h, 2) "Input must be a vector of length $(size(h, 2)), got $(length(v))"
    @inbounds for i in 1:length(v) ÷ 3
        a, b, c = srange(i)
        res[a] = -h.field[c] * v[b] + h.field[b] * v[c]
        res[b] = h.field[c] * v[a] - h.field[a] * v[c]
        res[c] = -h.field[b] * v[a] + h.field[a] * v[b]
    end
    return res
end
function hamiltonian(sys::ClassicalSpinSystem{T}, state::Vector{T}) where T
    h = ClassicalSpinHamiltonian(T; nspins=nv(sys.topology))
    return hamiltonian!(h, sys, state)
end
function hamiltonian!(h::ClassicalSpinHamiltonian{T}, sys::ClassicalSpinSystem{T}, state::Vector{T}) where T
    @assert length(state) == size(h, 2) "Input must be a vector of length $(size(h, 2)), got $(length(state))"
    fill!(h.field, zero(T))
    for (e, J) in zip(edges(sys.topology), sys.coupling)
        i, j = src(e), dst(e)
        h.field[srange(i)] .+= J .* state[srange(j)]
        h.field[srange(j)] .+= J .* state[srange(i)]
    end
    return h
end

function simulate!(spins::Vector{SpinVector{D, T}}, sys::ClassicalSpinSystem{T}; nsteps::Int, dt::T, checkpoint_steps::Int=10) where {D, T}
    # evolve with the Lie group integrator
    checkpoints = Vector{Vector{T}}()
    state = flatten(spins)
    h = hamiltonian(sys, state)
    for i in 1:nsteps
        state, _ = exponentiate(h, dt, state)
        # update effective Hamiltonian
        hamiltonian!(h, sys, state)
    end
    return unflatten!(spins, state)
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

function partite_edges(g::SimpleGraph)
    gdual = dual_graph(g)
    partite_edges = greedy_coloring(gdual)
    return [filter(==(i), partite_edges) for i in 1:maximum(partite_edges)]
end
