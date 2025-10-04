using Test, ImageProcessing, ImageProcessing.Images

@testset "demo_image" begin
    # Test valid images
    img = demo_image("cat.png")
    @test img isa AbstractArray
    @test size(img, 1) > 0 && size(img, 2) > 0
    
    # Test all available images
    for name in ["cat.png", "art.png", "amat.png"]
        img = demo_image(name)
        @test img isa Matrix{<:RGBA}
    end
    
    # Test error for invalid name
    @test_throws ArgumentError demo_image("invalid.png")
end

@testset "safe_convert" begin
    # Test basic conversion
    @test ImageProcessing.safe_convert(N0f8, 0.5) == N0f8(0.5)
    
    # Test clamping
    @test ImageProcessing.safe_convert(N0f8, 1.5) == N0f8(1.0)
    @test ImageProcessing.safe_convert(N0f8, -0.5) == N0f8(0.0)
end