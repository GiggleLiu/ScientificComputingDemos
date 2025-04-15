using SimpleLinearAlgebra: strassen
using Test

@testset "Strassen's Algorithm" begin
    A = rand(4, 4)
    B = rand(4, 4)
    C = strassen(A, B)
    @test C ≈ A * B
end