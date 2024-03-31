using DelimitedFiles

"""
    SpinGlassModel{T<:Real}

Annealing problem defined by coupling matrix of spins.
"""
struct SpinGlassModel{T<:Real}
    coupling::Matrix{T}
    function SpinGlassModel(coupling::Matrix{T}) where T
        size(coupling, 1) == size(coupling, 2) || throw(DimensionMismatch("input must be square matrix."))
        new{T}(coupling)
    end
end
num_spin(sap::SpinGlassModel) = size(sap.coupling, 1)

"""
    load_spinglass(filename::String) -> SpinGlassModel

Load spin glass problem from file.
"""
function load_spinglass(filename::String)
    data = readdlm(filename)
    is = Int.(view(data, :, 1)) .+ 1  #! @. means broadcast for the following functions, is here used correctly?
    js = Int.(view(data, :, 2)) .+ 1
    weights = data[:,3]
    num_spin = max(maximum(is), maximum(js))
    J = zeros(eltype(weights), num_spin, num_spin)
    for (i, j, weight) = zip(is, js, weights)
        J[i,j] = weight/2
        J[j,i] = weight/2
    end
    SpinGlassModel(J)
end

struct SpinConfig{Ts, Tf}
    config::Vector{Ts}
    field::Vector{Tf}
end

"""
    random_config(prob::AnnealingProblem) -> SpinConfig

Random spin configuration.
"""
function random_config(prob::SpinGlassModel)
    config = rand([-1,1], num_spin(prob))
    SpinConfig(config, prob.coupling*config)
end
     

"""
    anneal_singlerun!(config::AnnealingConfig, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing using Metropolis updates for the single run.

    * configuration that can be updated.
    * prob: problem with `energy`, `flip!` and `random_config` interfaces.
    * tempscales: temperature scales, which should be a decreasing array.
    * num_update_each_temp: the number of update in each temprature scale.

Returns (minimum cost, optimal configuration).
"""
function anneal_singlerun!(config, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)
    cost = energy(config, prob)
    
    opt_config = config
    opt_cost = cost
    for beta = 1 ./ tempscales
        @simd for _ = 1:num_update_each_temp  # single instriuction multiple data, see julia performance tips.
            proposal, ΔE = propose(config, prob)
            if exp(-beta*ΔE) > rand()  #accept
                flip!(config, proposal, prob)
                cost += ΔE
                if cost < opt_cost
                    opt_cost = cost
                    opt_config = config
                end
            end
        end
    end
    opt_cost, opt_config
end
 
"""
    anneal(nrun::Int, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)

Perform Simulated Annealing with multiple runs.
"""
function anneal(nrun::Int, prob, tempscales::Vector{Float64}, num_update_each_temp::Int)
    local opt_config, opt_cost
    for r = 1:nrun
        initial_config = random_config(prob)
        cost, config = anneal_singlerun!(initial_config, prob, tempscales, num_update_each_temp)
        if r == 1 || cost < opt_cost
            opt_cost = cost
            opt_config = config
        end
    end
    opt_cost, opt_config
end

"""
    energy(config::AnnealingConfig, ap::AnnealingProblem) -> Real

Get the cost of specific configuration.
"""
energy(config::SpinConfig, sap::SpinGlassModel) = sum(config.config'*sap.coupling*config.config)

"""
    propose(config::AnnealingConfig, ap::AnnealingProblem) -> (Proposal, Real)

Propose a change, as well as the energy change.
"""
@inline function propose(config::SpinConfig, ::SpinGlassModel)  # ommit the name of argument, since not used.
    ispin = rand(1:length(config.config))
    @inbounds ΔE = -config.field[ispin] * config.config[ispin] * 4 # 2 for spin change, 2 for mutual energy.
    ispin, ΔE
end

"""
    flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
@inline function flip!(config::SpinConfig, ispin::Int, sap::SpinGlassModel)
    @inbounds config.config[ispin] = -config.config[ispin]  # @inbounds can remove boundary check, and improve performance
    # update the field
    @simd for i=1:num_spin(sap)
        @inbounds config.field[i] += 2 * config.config[ispin] * sap.coupling[i,ispin]
    end
    config
end