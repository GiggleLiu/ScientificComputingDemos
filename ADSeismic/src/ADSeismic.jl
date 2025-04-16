module ADSeismic

using Enzyme

export solve_detector, treeverse_grad_detector,
    treeverse_solve_detector
export Glued, RK4, ODESolve, ODEStep,
    ODELog, checkpointed_neuralode
export AcousticPropagatorParams, solve

include("simulation.jl")
include("utils.jl")
#include("detector.jl")
include("treeverse.jl")

include("cuda.jl")

end
