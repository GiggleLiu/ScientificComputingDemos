using SpinDynamics
using Test

@testset "Spin dynamics" begin
    include("spindynamics.jl")
end

@testset "Visualize" begin
    include("visualize.jl")
end