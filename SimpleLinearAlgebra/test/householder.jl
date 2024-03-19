using Test, LinearAlgebra

@testset "householder property" begin
    v = randn(3)
    H = HouseholderMatrix(v)
    # symmetric
    @test H' ≈ H
    # reflexive
    @test H^2 ≈ I
    # orthogonal
    @test H' * H ≈ I
end

@testset "householder QR" begin
    A = randn(3, 3)
    Q = Matrix{Float64}(I, 3, 3)
    R = copy(A)
    householder_qr!(Q, R)
    @info R
    @test Q * R ≈ A
    @test Q' * Q ≈ I
end