module KernelPCA

using LinearAlgebra

export DataSets
export RBFKernel, PolyKernel, LinearKernel, kernelf, matrix, Point
export kpca, KPCAResult, ϕ

# include("pca.jl")
include("kernels.jl")
include("kpca.jl")
include("dataset.jl")

end
