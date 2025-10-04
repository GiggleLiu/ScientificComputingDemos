using Test, ImageProcessing, ImageProcessing.Images

@testset "FFT Compression" begin
    img = demo_image("cat.png")
    
    # Test basic compression
    compressed = fft_compress(img, 64, 64)
    @test compressed isa ImageProcessing.FFTCompressedImage
    
    # Test reconstruction
    reconstructed = toimage(RGBA{N0f8}, compressed)
    @test size(reconstructed) == size(img)
    
    # Test compression ratio
    ratio = compression_ratio(compressed)
    @test 0 < ratio < 1  # Should compress the image
    
    # Test lower rank (further compression)
    compressed_lower = lower_rank(compressed, 32, 32)
    @test compression_ratio(compressed_lower) < ratio
    
    # Test truncate_k function
    x = randn(ComplexF64, 10, 10)
    result = ImageProcessing.truncate_k(x, 5, 5)
    @test size(result) == (5, 5)
    
    # Test pad_zeros function
    result = ImageProcessing.pad_zeros(x, 20, 20)
    @test size(result) == (20, 20)
end