using GraphClustering
using Test

@testset "coo" begin
    include("coo.jl")
end

@testset "csc" begin
    include("csc.jl")
end

@testset "clustering" begin
    include("clustering.jl")
end