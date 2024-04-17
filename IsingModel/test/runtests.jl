using IsingModel
using Test

@testset "ising2d" begin
    include("ising2d.jl")
end

@testset "swendsen_wang" begin
    include("swendsen_wang.jl")
end