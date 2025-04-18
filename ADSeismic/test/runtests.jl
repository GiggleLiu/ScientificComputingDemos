using ADSeismic, CUDA
using Test

@testset "simulation" begin
    include("simulation.jl")
end

@testset "treeverse" begin
    include("treeverse.jl")
end

if CUDA.functional()
    @testset "cuda" begin
        include("cuda.jl")
    end
end
