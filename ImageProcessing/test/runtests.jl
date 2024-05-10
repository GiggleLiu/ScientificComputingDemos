using ImageProcessing
using Test

@testset "fft" begin
    include("fft.jl")
end

@testset "pca" begin
    include("pca.jl")
end

@testset "polymul" begin
    include("polymul.jl")
end
