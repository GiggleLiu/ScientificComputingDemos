module IsingModel

using DelimitedFiles

export SpinModel, mcstep!, SimulationResult, energy, measure!, simulate!

include("ising2d.jl")

end
