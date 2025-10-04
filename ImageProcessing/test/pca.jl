using Test, ImageProcessing, ImageProcessing.Images

@testset "SVD Compression" begin
    img = demo_image("cat.png")
    
    # Test basic compression
    compressed = svd_compress(img, 10)
    @test compressed isa ImageProcessing.SVDCompressedImage
    
    # Test reconstruction
    reconstructed = toimage(RGBA{N0f8}, compressed)
    @test size(reconstructed) == size(img)
    
    # Test compression ratio
    ratio = compression_ratio(compressed)
    @test 0 < ratio < 1  # Should compress the image
    
    # Test lower rank
    compressed_lower = lower_rank(compressed, 5)
    @test compression_ratio(compressed_lower) < ratio
    
    # Test error handling
    @test_throws AssertionError svd_compress(img, 0)
end