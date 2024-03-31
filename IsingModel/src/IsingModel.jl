module IsingModel

using DelimitedFiles

export IsingSpinModel, mcstep!, SimulationResult, energy, measure!, simulate!
export SpinGlassModel, load_spinglass, anneal, random_config

include("ising2d.jl")
include("spinglass.jl")

end
