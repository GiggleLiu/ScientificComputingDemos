module IsingModel

using DelimitedFiles

export IsingModel, mcstep!, SimulationResult, energy, measure!, simulate!

include("ising2d.jl")

end
