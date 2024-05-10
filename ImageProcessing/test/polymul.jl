using Test, ImageProcessing

@testset "fast_polymul" begin
    @test fast_polymul([1, 2, 3], [4, 5, 6]) ≈ [4, 13, 28, 27, 18]
end