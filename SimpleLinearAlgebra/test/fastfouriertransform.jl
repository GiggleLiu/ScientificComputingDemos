using LinearAlgebra, Test

@testset "fft" begin
    x = randn(ComplexF64, 8)
    @test fft!(copy(x)) ≈ dft_matrix(8) * x
end

@testset "ifft" begin
    x = randn(ComplexF64, 8)
    @test ifft!(copy(x)) ≈ inv(dft_matrix(8)) * x
end