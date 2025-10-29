using TensorRenormalizationGroup
using Test

@testset "trg" begin
    include("trg.jl")
end

@testset "autodiff" begin
    include("autodiff.jl")
end