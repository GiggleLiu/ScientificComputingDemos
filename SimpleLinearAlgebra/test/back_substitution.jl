using Test, LinearAlgebra

@testset "back substitution" begin
    # create a random lower triangular matrix
    l = LinearAlgebra.tril(randn(4, 4))
    # target vector
    b = randn(4)
    # solve the linear equation with our algorithm
    x = back_substitution!(l, copy(b))
    @test l * x ≈ b

    # The Julia's standard library `LinearAlgebra` contains a native implementation.
    x_native = LowerTriangular(l) \ b
    @test l * x_native ≈ b
end
