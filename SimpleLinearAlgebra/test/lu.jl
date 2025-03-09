using Test, LinearAlgebra
using SimpleLinearAlgebra: classical_gram_schmidt, modified_gram_schmidt!, lufact_pivot!, forward_substitution!, lufact

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

@testset "lufact with pivot" begin
    n = 5
    A = randn(n, n)
    L, U, P = lufact_pivot!(copy(A))
    pmat = zeros(Int, n, n)
    setindex!.(Ref(pmat), 1, 1:n, P)
    @test L ≈ lu(A).L
    @test U ≈ lu(A).U
    @test pmat * A ≈ L * U
end

@testset "forward substitution" begin
    # create a random lower triangular matrix
    l = LinearAlgebra.tril(randn(4, 4))
    # target vector
    b = randn(4)
    # solve the linear equation with our algorithm
    x = forward_substitution!(l, copy(b))
    @test l * x ≈ b

    # The Julia's standard library `LinearAlgebra` contains a native implementation.
    x_native = LowerTriangular(l) \ b
    @test l * x_native ≈ b
end


@testset "LU factorization" begin
    a = randn(4, 4)
    L, U = lufact(a)
    @test istril(L)
    @test istriu(U)
    @test L * U ≈ a
end