module Spinglass

using DelimitedFiles, Graphs, GenericTensorNetworks
export load_spinglass, random_config, anneal
export SpinConfig, SpinglassModel
export sg_gadget_and, ground_states, sg_gadget_not, sg_gadget_or, sg_gadget_arraymul, truth_table

include("simulated_annealing.jl")
include("logic_gates.jl")

end
