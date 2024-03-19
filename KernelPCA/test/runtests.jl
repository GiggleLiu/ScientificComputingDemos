using KernelPCA
using Test

@testset "kernels" begin
    include("kernels.jl")
end

@testset "kpca" begin
    include("kpca.jl")
end