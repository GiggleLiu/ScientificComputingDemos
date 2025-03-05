# source: https://lampz.tugraz.at/~hadley/ss1/phonons/1d/1dphonons.php
# the abstract type for the Hamiltonian system
abstract type AbstractHamiltonianSystem{D} end

"""
    SpringModel{T, D} <: AbstractHamiltonianSystem{D}

The spring model is a simple model of a atoms connected by springs.

### Fields
- `r0` is the position of the atoms
- `dr` is the offset of the atoms
- `v` is the velocity of the atoms
- `topology` is the topology of the spring system, which is a graph
- `stiffness` is the stiffness of the springs defined on the edges
- `mass` is the mass defined on the atoms
"""
struct SpringModel{T, D} <: AbstractHamiltonianSystem{D}
    r0::Vector{Point{D, T}}   # the position of the atoms
    dr::Vector{Point{D, T}}   # the offset of the atoms
    v::Vector{Point{D, T}}   # the velocity of the atoms
    topology::SimpleGraph{Int}   # the topology of the spring system
    stiffness::Vector{T}  # stiffness of the springs defined on the edges
    mass::Vector{T}       # defined on the atoms
    function SpringModel(r0::Vector{Point{D, T}}, dr::Vector{Point{D,T}}, v::Vector{Point{D, T}}, topology::SimpleGraph{Int}, stiffness::Vector{T}, mass::Vector{T}) where {T, D}
        @assert length(r0) == length(dr) == length(v) == length(stiffness) == length(mass)
        new{T, D}(r0, dr, v, topology, stiffness, mass)
    end
end

# get the coordinates of the atoms
coordinate(sys::SpringModel) = sys.r0 .+ sys.dr
coordinate(sys::SpringModel, i::Int) = sys.r0[i] + sys.dr[i]

# get the offset of the atoms
offset(sys::SpringModel) = sys.dr
offset(sys::SpringModel, i::Int) = sys.dr[i]

# get the velocity of the atoms
velocity(sys::SpringModel) = sys.v
velocity(sys::SpringModel, i::Int) = sys.v[i]

# get the mass of the atoms
mass(sys::SpringModel) = sys.mass
mass(sys::SpringModel, i::Int) = sys.mass[i]

# update the offset of the atoms
function offset_coordinate!(sys::SpringModel, i::Int, val)
    sys.dr[i] += val
end

# update the velocity of the atoms
function offset_velocity!(sys::SpringModel, i::Int, val)
    sys.v[i] += val
end

# get the number of atoms
Base.length(sys::SpringModel) = length(sys.r0)

# update the acceleration of the atoms
function update_acceleration!(a::AbstractVector{Point{D, T}}, bds::SpringModel) where {D, T}
    @assert length(a) == length(bds)
    fill!(a, zero(Point{D, T}))
    @inbounds for (k, e) in zip(bds.stiffness, edges(bds.topology))
        i, j = src(e), dst(e)
        f = k * (offset(bds, i) - offset(bds, j))
        a[j] += f / mass(bds, j)
        a[i] -= f / mass(bds, i)
    end
    return a
end

# create a spring chain with n atoms
function spring_chain(offsets::Vector{<:Real}, stiffness::Real, mass::Real; periodic::Bool)
    n = length(offsets)
    r = Point.(0.0:n-1)
    dr = Point.(Float64.(offsets))
    v = fill(Point(0.0), n)
    topology = path_graph(n)
    periodic && add_edge!(topology, n, 1)
    return SpringModel(r, dr, v, topology, fill(stiffness, n), fill(mass, n))
end

# the eigensystem of the chain: (K - ω^2 M) v = 0, where K is the stiffness matrix, M is the mass matrix
struct EigenSystem{T}
    K::Matrix{T}
    ms::Vector{T}
end
# coordinate(e::EigenSystem, i::Int) = e.K[i, i]

# stiffness and mass can be either a scalar or a vector
function eigensystem(spr::SpringModel{T}) where T
    eigensystem(T, spr.topology, spr.stiffness, spr.mass)
end
function eigensystem(::Type{T}, g::SimpleGraph, stiffness, mass) where T
    n = nv(g)
    M = zeros(T, n)
    M .= mass
    K = zeros(T, n, n)
    for (s, edg) in zip(stiffness, edges(g))
        i, j = src(edg), dst(edg)
        # site i feels a force: stiffness * (x_i - x_j)
        K[i, i] += s
        K[i, j] -= s
        # site j feels a force: -stiffness * (x_i - x_j)
        K[j, j] += s
        K[j, i] -= s
    end
    return EigenSystem(K, M)
end

"""
    EigenModes{T}

The eigenmodes of the spring model.

### Fields
- `frequency` is the frequency of the eigenmodes
- `modes` is the eigenmodes, which stores the amplitude of the eigenmodes
"""
struct EigenModes{T}
    frequency::Vector{T}
    modes::Matrix{T}
end

# compute the eigenmodes of the spring model
function eigenmodes(e::EigenSystem{T}) where T
    vals, vecs = eigen(e.K)
    frequency = sqrt.(vals ./ e.ms)
    return EigenModes(frequency, vecs)
end

# compute the wave amplitude at a given time
waveat(es::EigenModes, idx::Int, t; phi0=0.0) = ut(es.frequency[idx], t, es.modes[:,idx]; phi0)
function waveat(es::EigenModes, u0::AbstractVector, ts::AbstractVector)
    coeffs = es.modes' * u0
    map(ts) do t
        # a linear combination of the eigenmodes at time t
        sum(coeffs[idx] * waveat(es, idx, t) for idx in 1:length(es.frequency))
    end
end
ut(omega::Real, t::Real, A0::AbstractVector; phi0=0.0) = real(exp(-im * omega * t + phi0) * A0)