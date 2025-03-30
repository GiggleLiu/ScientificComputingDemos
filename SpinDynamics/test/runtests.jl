using SpinDynamics
using Test

@testset "Spin dynamics" begin
    include("simulation.jl")
end

@testset "Simulated bifurcation" begin
    include("simulated_bifurcation.jl")
end