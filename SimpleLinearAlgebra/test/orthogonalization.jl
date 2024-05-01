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