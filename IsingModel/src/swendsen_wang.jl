# A stack with fixed size
mutable struct FixedSizedStack{T}
    const data::Vector{T}   # data
    top::Int                # top index
end
FixedSizedStack{T}(n::Int) where T = FixedSizedStack{T}(Vector{T}(undef, n), 0)
Base.isempty(stack::FixedSizedStack) = stack.top == 0
Base.length(stack::FixedSizedStack) = stack.top
reset!(stack::FixedSizedStack) = stack.top = 0
# push a value to the stack
function Base.push!(stack::FixedSizedStack, x)
    @boundscheck checkbounds(stack.data, stack.top + 1)
    stack.top += 1
    @inbounds stack.data[stack.top] = x
end
# pop a value from the stack
function Base.pop!(stack::FixedSizedStack)
    @boundscheck checkbounds(stack.data, stack.top)
    stack.top -= 1
    return @inbounds stack.data[stack.top + 1]
end

# The Swendsen-Wang model for efficient cluster Monte Carlo simulation
struct SwendsenWangModel{RT} <: AbstractSpinModel
    l::Int                  # lattice size
    h::RT                   # magnetic field
    beta::RT                # inverse temperature 1/T
    prob::RT                # flip probability
    neigh::Matrix{Int}      # neighbors
    bondspin::Matrix{Int}   # map bonds to spins
    spinbond::Matrix{Int}   # map spins to bonds
end
function SwendsenWangModel(l::Int, h::RT, beta::RT) where RT
    neigh = lattice(l)
    bondspin, spinbond = spinbondmap(neigh)
    prob = 1 - exp(-2 * beta)
    SwendsenWangModel(l, h, beta, prob, neigh, bondspin, spinbond)
end
num_spin(model::SwendsenWangModel) = model.l^2
energy(model::SwendsenWangModel, spin) = ferromagnetic_energy(model.neigh, model.h, spin)

# Constructs the tables corresponding to the lattice structure (here 2D square)
# - the sites (spins) are labeled 1,...,N. The bonds are labeled 1,...,2N.
# - neighbor[i,s] = i:th neighbor sote of site s (i=1,2,3,4)
# - bondspin[i,b] = i:th site connected to bond b (i=1,2)
# - spinbond[i,s] = i:th bond connected to spin s (i=1,2,3,4)
function spinbondmap(neighbor)
    nn = size(neighbor, 2)
    bondspin = zeros(Int, 2, 2*nn)
    spinbond = zeros(Int, 4, nn)
    for s0 = 1:nn
        # map bonds to spin
        bondspin[1, s0] = s0
        bondspin[2, s0] = neighbor[1, s0]
        bondspin[1, nn+s0] = s0
        bondspin[2, nn+s0] = neighbor[2, s0]
        # map spins to bonds
        spinbond[1, s0] = s0
        spinbond[2, s0] = nn + s0
        spinbond[3, neighbor[1, s0]] = s0
        spinbond[4, neighbor[2, s0]] = nn + s0
    end
    return bondspin, spinbond
end

# The spin configuration and the bond configuration for the Swendsen-Wang algorithm
struct SwendsenWangConfig
    spin::Matrix{Int}               # spin configuration   
    bond::Vector{Bool}              # bond configuration
    cflag::Vector{Bool}             # for caching, visited sites are marked as false
    cstack::FixedSizedStack{Int}    # for caching, stack for cluster construction
end
function SwendsenWangConfig(spin)
    nn = length(spin)
    SwendsenWangConfig(spin, zeros(Bool, 2nn), trues(nn), FixedSizedStack{Int}(nn))
end

# Generates a valid bond configuration, given a spin configuration
function castbonds!(config::SwendsenWangConfig, model::SwendsenWangModel)
    for b in eachindex(config.bond)
        # NOTE: only flips parallel spins
        config.bond[b] = config.spin[model.bondspin[1,b]] == config.spin[model.bondspin[2,b]] && rand() < model.prob
    end 
    return config
end

# Constructs all the clusters and flips each of them with probability 1/2
function flipclusters!(config::SwendsenWangConfig, model::SwendsenWangModel)
    cstack, cflag = config.cstack, config.cflag
    neighbor, spinbond = model.neigh, model.spinbond
    cflag .= true  # visited sites are marked as false
    @inbounds for cseed = 1:length(cflag)    # construct clusters until all sites visited (then cseed=0)
        cflag[cseed] || continue   # skip visited sites

        reset!(cstack)
        push!(cstack, cseed)
        cflag[cseed] = false

        doflip = rand() < 0.5 # flip a cluster with probability 1/2
        doflip && (config.spin[cseed] = -config.spin[cseed])
        while !isempty(cstack)
            s0 = pop!(cstack)
            for i=1:4  # for each neighbor of s0
                s1 = neighbor[i,s0]
                if config.bond[spinbond[i,s0]] && cflag[s1]
                    push!(cstack, s1)
                    cflag[s1] = false
                    doflip && (config.spin[s1] = -config.spin[s1])
                end 
            end
        end
    end
    return nothing
end

# Monte Carlo step
function mcstep!(model::SwendsenWangModel, config::SwendsenWangConfig)
    castbonds!(config, model)
    flipclusters!(config, model)
end

# Simulate the Ferromagnetic spin model with the Swendsen-Wang method
function simulate!(model::SwendsenWangModel, spin; nsteps_heatbath, nsteps_eachbin, nbins)
    @assert length(spin) == model.l^2
    config = SwendsenWangConfig(spin)

    for _ = 1:nsteps_heatbath
        mcstep!(model, config)
    end

    result = SimulationResult(nbins, nsteps_eachbin)
    for j = 1:nbins
        result.current_bin[] = j
        for _ = 1:nsteps_eachbin
            mcstep!(model, config)
            measure!(result, model, config.spin)
        end
    end
    return result
end