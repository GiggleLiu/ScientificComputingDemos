using GraphClustering
using Test

@testset "coo" begin
    include("coo.jl")
end

@testset "csc" begin
    include("csc.jl")
end

@testset "visualization" begin
    include("visualization.jl")
end