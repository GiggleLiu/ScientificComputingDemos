# required interfaces: num_spin, energy
abstract type AbstractSpinModel end

# the Ising model
struct IsingSpinModel{RT} <: AbstractSpinModel
    l::Int                  # lattice size
    h::RT                   # magnetic field
    beta::RT                # inverse temperature 1/T

    pflp::NTuple{10, RT}    # precompiled flip probability
    neigh::Matrix{Int}      # neighbors, neigh[1-4, i]
end
function IsingSpinModel(l::Int, h::RT, beta::RT) where RT
    pflp = ([exp(-2*s*(i + h) * beta) for s=-1:2:1, i in -4:2:4]...,)
    neigh = lattice(l)
    IsingSpinModel(l, h, beta, pflp, neigh)
end

# Constructs a list neigh[1:4,1:nn] of neighbors of each site
function lattice(ll)
    lis = LinearIndices((ll, ll))
    return reshape([lis[mod1(ci.I[1]+di, ll), mod1(ci.I[2]+dj, ll)]
        for (di, dj) in ((1, 0), (0, 1), (-1, 0), (0, -1)),
            ci in CartesianIndices((ll, ll))],
            4, ll*ll)
end

# Returns the number of spins
num_spin(model::IsingSpinModel) = model.l^2

# Returns the energy of the spin configuration
energy(model::IsingSpinModel, spin) = ferromagnetic_energy(model.neigh, model.h, spin)
function ferromagnetic_energy(neigh::AbstractMatrix, h::Real, spin::AbstractMatrix)
    @boundscheck size(neigh) == (4, length(spin))
    sum(1:length(spin)) do i
        s = spin[i]
        - s * (spin[neigh[1, i]] + spin[neigh[2, i]] + h)
    end
end

# Returns the precompiled flip probability
@inline function pflip(model::IsingSpinModel, s::Integer, field::Integer)
    return @inbounds model.pflp[(field + 5) + (1 + s) >> 1]
end

# Monte Carlo step
function mcstep!(model::IsingSpinModel, spin)
    nn = num_spin(model)
    @inbounds for _ = 1:nn
        s = rand(1:nn)
        field = spin[model.neigh[1, s]] + spin[model.neigh[2, s]] + spin[model.neigh[3, s]] + spin[model.neigh[4, s]]
        if rand() < pflip(model, spin[s], field)
           spin[s] = -spin[s]
        end
    end    
end

# A data structure to store the simulation result
struct SimulationResult{RT}
    nbins::Int                      # number of bins
    nsteps_eachbin::Int             # number of steps in each bin
    energy::Vector{RT}              # energy/spin
    energy2::Vector{RT}             # (energy/spin)^2
    m::Vector{RT}                   # |m|
    m2::Vector{RT}                  # m^2
    m4::Vector{RT}                  # m^4
end
SimulationResult(nbins, nsteps_eachbin) = SimulationResult(nbins, nsteps_eachbin, zeros(nbins), zeros(nbins), zeros(nbins), zeros(nbins), zeros(nbins))

# Measures the energy and magnetization
# j: the index of the bin
# k: the index of the step
function measure!(result::SimulationResult, model::AbstractSpinModel, spin, j::Int)
    @boundscheck checkbounds(result.energy, j)
    m = sum(spin)
    e = energy(model, spin)
    n = num_spin(model)
    @inbounds result.energy[j] += e/n
    @inbounds result.energy2[j] += (e/n)^2
    @inbounds result.m[j] += abs(m/n)
    @inbounds result.m2[j] += (m/n)^2
    @inbounds result.m4[j] += (m/n)^4
    return m/n, e/n
end

# Simulates the Ising model
function simulate!(model::AbstractSpinModel, spin; nsteps_heatbath::Int, nsteps_eachbin::Int, nbins::Int, taumax::Int)
    # heat bath
    for _ = 1:nsteps_heatbath
        mcstep!(model, spin)    
    end
    result = SimulationResult(nbins, nsteps_eachbin)

    mvec = zeros(taumax)  # a sequence of m, for computing the autocorrelation time
    m_mean, m_mean2, m_corr_mean = 0.0, zeros(taumax), zeros(taumax)
    k = 0
    for j=1:nbins, _ in 1:nsteps_eachbin
        k += 1
        mcstep!(model, spin)
        m, _ = measure!(result, model, spin, j)

        # update mvec
        circshift!(mvec, -1)
        mvec[end] = abs(m)

        # update m_mean, m_mean2, m_corr_mean
        if k > taumax
            m_mean += mvec[1]
            m_mean2 .+= (mvec[1]^2 .+ mvec .^ 2) ./ 2
            m_corr_mean .+= mvec[1] .* mvec
        end
    end
    m_mean, m_mean2, m_corr_mean = m_mean / (k - taumax), m_mean2 / (k - taumax), m_corr_mean / (k - taumax)
    return result, (m_corr_mean .- m_mean^2) ./ (m_mean2 .- m_mean^2)
end

# Write the simulation result to a file
function Base.write(filename::String, result::SimulationResult)
    mdata = hcat(result.energy, result.energy2, result.m, result.m2, result.m4) / result.nsteps_eachbin
    writedlm(filename, mdata)
end