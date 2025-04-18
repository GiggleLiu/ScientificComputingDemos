module ADSeismic

using Enzyme
using TreeverseAlgorithm
using LinearAlgebra

export AcousticPropagatorParams, solve
export treeverse, treeverse_gradient

include("simulation.jl")
include("utils.jl")
#include("detector.jl")
include("treeverse.jl")

include("cuda.jl")

end
