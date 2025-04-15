using Test, LinearAlgebra
using SimpleLinearAlgebra: classical_gram_schmidt, modified_gram_schmidt, givens_qr!, householder_qr!, HouseholderMatrix

@testset "givens QR" begin
    n = 3
    A = randn(n, n)
    R = copy(A)
    Q, R = givens_qr!(Matrix{Float64}(I, n, n), R)
    @test Q * R ≈ A
    @test Q * Q' ≈ I
    @info R
end


@testset "householder property" begin
    v = randn(3)
    H = HouseholderMatrix(v)
    A = randn(3, 3)
    @test H * A ≈ Matrix(H) * A
    @test A * H ≈ A * Matrix(H)
    CA = copy(A)
    mul!(CA, H, CA)
    @test CA ≈ H * A
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