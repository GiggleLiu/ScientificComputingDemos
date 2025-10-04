using Test, ImageProcessing, ImageProcessing.Images

@testset "Utils" begin
    include("utils.jl")
end

@testset "SVD Compression" begin
    include("pca.jl")
end

@testset "FFT Compression" begin
    include("fft.jl")
end