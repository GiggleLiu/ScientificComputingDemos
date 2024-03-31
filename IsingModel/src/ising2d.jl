struct IsingSpinModel{RT}
    l::Int
    h::RT
    beta::RT
    pflp::NTuple{10, RT}
    neigh::Matrix{Int}
end
function IsingSpinModel(l::Int, h::RT, beta::RT) where RT
    pflp = ([exp(-2*s*(i + h) * beta) for s=-1:2:1, i in -4:2:4]...,)
    neigh = lattice(l)
    IsingSpinModel(l, h, beta, pflp, neigh)
end
# Constructs a list neigh[1:4,1:nn] of neighbors of each site
function lattice(ll)
    lis = LinearIndices((ll, ll))
    return reshape([lis[mod1(ci.I[1]+di, ll), mod1(ci.I[2]+dj, ll)] for (di, dj) in ((1, 0), (0, 1), (-1, 0), (0, -1)), ci in CartesianIndices((ll, ll))], 4, ll*ll)
end

# Performs one MC sweep. This version computes the neighbors on the fly.
@inline function pflip(model::IsingSpinModel, s::Integer, field::Integer)
    return @inbounds model.pflp[(field + 5) + (1 + s) >> 1]
end

function mcstep!(model::IsingSpinModel, spin)
    nn = model.l^2
    @inbounds for _ = 1:nn
        s = rand(1:nn)
        field = spin[model.neigh[1, s]] + spin[model.neigh[2, s]] + spin[model.neigh[3, s]] + spin[model.neigh[4, s]]
        if rand() < pflip(model, spin[s], field)
           spin[s] = -spin[s]
        end
    end    
end

# Measures physical quantities and accumulates them in mdata
struct SimulationResult{RT}
    nbins::Int
    nsteps_eachbin::Int
    current_bin::Base.RefValue{Int}
    energy::Vector{RT}  # energy/spin
    energy2::Vector{RT}  # (energy/spin)^2
    m::Vector{RT}  # |m|
    m2::Vector{RT}  # m^2
    m4::Vector{RT}  # m^4
end
SimulationResult(nbins, nsteps_eachbin) = SimulationResult(nbins, nsteps_eachbin, Ref(0), zeros(nbins), zeros(nbins), zeros(nbins), zeros(nbins), zeros(nbins))

function energy(model::IsingSpinModel, spin)
    sum(1:model.l^2) do i
        s = spin[i]
        - s * (spin[model.neigh[1, i]] + spin[model.neigh[2, i]] + model.h)
    end
end

function measure!(result::SimulationResult, model::IsingSpinModel, spin)
    @boundscheck checkbounds(result.energy, result.current_bin[])
    m = sum(spin)
    e = energy(model, spin)
    n = model.l^2
    k = result.current_bin[]
    @inbounds result.energy[k] += e/n
    @inbounds result.energy2[k] += (e/n)^2
    @inbounds result.m[k] += abs(m/n)
    @inbounds result.m2[k] += (m/n)^2
    @inbounds result.m4[k] += (m/n)^4
end

function simulate!(model::IsingSpinModel, spin; nsteps_heatbath, nsteps_eachbin, nbins)
    # heat bath
    for _ = 1:nsteps_heatbath
        mcstep!(model, spin)    
    end
    result = SimulationResult(nbins, nsteps_eachbin)
    for j=1:nbins
        result.current_bin[] = j
        for _ = 1:nsteps_eachbin
            mcstep!(model, spin)
            measure!(result, model, spin)
        end
    end
    result
end

# Writes the bin averages to the file res.dat, writes a message to 'log.log'
function Base.write(filename::String, result::SimulationResult)
    mdata = hcat(result.energy, result.energy2, result.m, result.m2, result.m4) / result.nsteps_eachbin
    writedlm(filename, mdata)
end