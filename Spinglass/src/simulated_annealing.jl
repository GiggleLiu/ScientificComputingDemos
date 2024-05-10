"""
    load_spinglass(filename::String) -> SpinGlass

Load spin glass problem from file.
"""
function load_spinglass(filename::String)
    data = readdlm(filename)
    is = Int.(view(data, :, 1)) .+ 1  #! @. means broadcast for the following functions, is here used correctly?
    js = Int.(view(data, :, 2)) .+ 1
    num_spin = max(maximum(is), maximum(js))
    SpinGlass(num_spin, collect.(zip(is, js)), data[:,3])
end

struct SpinGlassSA{T, MT<:AbstractMatrix{T}}
    coupling::MT
end
function SpinGlassSA(sg::SpinGlass)
    @assert all(x->length(x)==2, sg.cliques) "cliques should only contain quadratic terms"
    coupling = zeros(sg.n, sg.n)
    for ((i, j), weight) in zip(sg.cliques, sg.weights)
        coupling[i, j] = weight/2
        coupling[j, i] = weight/2
    end
    SpinGlassSA(coupling)
end
num_spin(prob::SpinGlassSA) = size(prob.coupling, 1)

struct SpinConfig{Ts, Tf}
    config::Vector{Ts}
    field::Vector{Tf}
end

"""
    random_config(prob::AnnealingProblem) -> SpinConfig

Random spin configuration.
"""
function random_config(prob::SpinGlassSA)
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
    
    opt_config = deepcopy(config)
    opt_cost = cost
    for beta = 1 ./ tempscales
        @simd for _ = 1:num_update_each_temp  # single instriuction multiple data, see julia performance tips.
            proposal, ΔE = propose(config, prob)
            if exp(-beta*ΔE) > rand()  #accept
                flip!(config, proposal, prob)
                cost += ΔE
                if cost < opt_cost
                    opt_cost = cost
                    opt_config = deepcopy(config)
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
anneal(nrun::Int, prob::SpinGlass, tempscales::Vector{Float64}, num_update_each_temp::Int) = anneal(nrun, SpinGlassSA(prob), tempscales, num_update_each_temp)
function anneal(nrun::Int, prob::SpinGlassSA, tempscales::Vector{Float64}, num_update_each_temp::Int)
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
energy(config::SpinConfig, sap::SpinGlassSA) = sum(config.config'*sap.coupling*config.config)

"""
    propose(config::AnnealingConfig, ap::AnnealingProblem) -> (Proposal, Real)

Propose a change, as well as the energy change.
"""
@inline function propose(config::SpinConfig, ::SpinGlassSA)  # ommit the name of argument, since not used.
    ispin = rand(1:length(config.config))
    @inbounds ΔE = -config.field[ispin] * config.config[ispin] * 4 # 2 for spin change, 2 for mutual energy.
    ispin, ΔE
end

"""
    flip!(config::AnnealingConfig, ispin::Proposal, ap::AnnealingProblem) -> SpinConfig

Apply the change to the configuration.
"""
@inline function flip!(config::SpinConfig, ispin::Int, sap::SpinGlassSA)
    @inbounds config.config[ispin] = -config.config[ispin]  # @inbounds can remove boundary check, and improve performance
    # update the field
    @simd for i=1:num_spin(sap)
        @inbounds config.field[i] += 2 * config.config[ispin] * sap.coupling[i,ispin]
    end
    config
end
