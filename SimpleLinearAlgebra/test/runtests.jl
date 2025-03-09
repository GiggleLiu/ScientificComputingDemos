using SimpleLinearAlgebra
using Test

@testset "qr.jl" begin
    include("qr.jl")
end

@testset "lu.jl" begin
    include("lu.jl")
end

@testset "fft.jl" begin
    include("fft.jl")
end