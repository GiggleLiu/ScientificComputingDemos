using SpinDynamics
using Test

@testset "Spin dynamics" begin
    include("simulation.jl")
end

@testset "Visualize" begin
    include("visualize.jl")
end