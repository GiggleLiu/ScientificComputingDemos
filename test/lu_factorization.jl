using Test, LinearAlgebra

lufact(a::AbstractMatrix) = lufact!(copy(a))

@testset "naive LU factorization" begin
    A = [1 2 2; 4 4 2; 4 6 4]
    L, U = lufact_naive!(copy(A))
    @test L * U ≈ A
end

@testset "LU factorization" begin
    a = randn(4, 4)
    L, U = lufact(a)
    @test istril(L)
    @test istriu(U)
    @test L * U ≈ a
end