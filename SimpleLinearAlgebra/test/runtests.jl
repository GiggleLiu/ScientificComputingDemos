using SimpleLinearAlgebra
using Test

@testset "strassen.jl" begin
    include("strassen.jl")
end

@testset "qr.jl" begin
    include("qr.jl")
end

@testset "lu.jl" begin
    include("lu.jl")
end

@testset "fft.jl" begin
    include("fft.jl")
end