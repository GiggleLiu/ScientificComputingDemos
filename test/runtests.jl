using SimpleLinearAlgebra
using Test

@testset "back_substitution.jl" begin
    include("back_substitution.jl")
end

@testset "lu_factorization.jl" begin
    include("lu_factorization.jl")
end

@testset "lu_factorization_partialpivoting.jl" begin
    include("lu_factorization_partialpivoting.jl")
end

@testset "householder.jl" begin
    include("householder.jl")
end

@testset "qr_factorization.jl" begin
    include("qr_factorization.jl")
end

@testset "orthogonalization.jl" begin
    include("orthogonalization.jl")
end

@testset "fouriertransform.jl" begin
    include("fouriertransform.jl")
end

