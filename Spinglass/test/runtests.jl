using Spinglass
using Test

@testset "simulated_annealing" begin
    include("simulated_annealing.jl")
end

@testset "logic_gates" begin
    include("logic_gates.jl")
end

@testset "dynamics" begin
    include("dynamics.jl")
end

@testset "Spin dynamics" begin
    include("spindynamics.jl")
end