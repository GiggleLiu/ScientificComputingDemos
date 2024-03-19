using Test
using MyFirstPackage

@testset "lorenz" begin
    include("lorenz.jl")
end