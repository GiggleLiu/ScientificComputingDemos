using Test, LinearAlgebra

@testset "classical GS" begin
    n = 10
    A = randn(n, n)
    Q, R = classical_gram_schmidt(A)
    @test Q * R ≈ A
    @test Q * Q' ≈ I
    @info R
end

@testset "modified GS" begin
    n = 10
    A = randn(n, n)
    Q, R = modified_gram_schmidt!(copy(A))
    @test Q * R ≈ A
    @test Q * Q' ≈ I
    @info R
end

let
    n = 100
    A = randn(n, n)
    Q1, R1 = classical_gram_schmidt(A)
    Q2, R2 = modified_gram_schmidt!(copy(A))
    @info norm(Q1' * Q1 - I)
    @info norm(Q2' * Q2 - I)
end