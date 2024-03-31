module IsingModel

using DelimitedFiles

export IsingSpinModel, mcstep!, SimulationResult, energy, measure!, simulate!, num_spin
export SpinGlassModel, load_spinglass, anneal, random_config
export SwendsenWangModel, castbonds, SwendsenWangConfig

include("ising2d.jl")
include("swendsen_wang.jl")
include("spinglass.jl")

end
