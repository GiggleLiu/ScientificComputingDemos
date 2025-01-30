module SimpleTensorNetwork

using OMEinsum, OMEinsum.LinearAlgebra
using Graphs
using CairoMakie
using NPZ

export Spinglass, TensorNetwork, OptimizedTensorNetwork
export generate_tensor_network, optimize_tensornetwork, partition_function
export partition_function_exact

include("tucker.jl")
include("spinglass.jl")
include("sampling.jl")

end
