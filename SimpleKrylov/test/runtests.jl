using SimpleKrylov
using Test

@testset "lanczos" begin
    include("lanczos.jl")
end

@testset "arnoldi" begin
    include("arnoldi.jl")
end