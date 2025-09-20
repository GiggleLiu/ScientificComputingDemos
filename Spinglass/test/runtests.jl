using Spinglass
using Test, CUDA

@testset "simulated_annealing" begin
    include("simulated_annealing.jl")
end

@testset "logic_gates" begin
    include("logic_gates.jl")
end

@testset "dynamics" begin
    include("dynamics.jl")
end

if CUDA.functional()
    @testset "CUDAExt" begin
        include("CUDAExt.jl")
    end
end