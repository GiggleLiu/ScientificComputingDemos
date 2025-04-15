using SimpleKrylov
using Test

@testset "coo" begin
    include("coo.jl")
end

@testset "csc" begin
    include("csc.jl")
end

@testset "lanczos" begin
    include("lanczos.jl")
end

@testset "arnoldi" begin
    include("arnoldi.jl")
end