using Test
using MyFirstPackage

@testset "lorenz" begin
    include("lorenz.jl")
end

@testset "fluid" begin
    include("fluid.jl")
end