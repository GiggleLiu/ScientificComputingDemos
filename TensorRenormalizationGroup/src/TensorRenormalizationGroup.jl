module TensorRenormalizationGroup

using OMEinsum  # for tensor contractions
using Zygote    # for automatic differentiation
using LinearAlgebra

include("trg.jl")
include("autodiff.jl")
include("models.jl")
include("utils.jl")

export trg, Ising, model_tensor, num_grad

end
