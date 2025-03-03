using SpringSystem
using Test

@testset "point" begin
    include("point.jl")
end

@testset "leapfrog" begin
    include("leapfrog.jl")
end

@testset "chain" begin
    include("chain.jl")
end

