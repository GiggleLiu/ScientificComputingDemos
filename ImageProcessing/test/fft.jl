using Test, ImageProcessing, ImageProcessing.Images

@testset "truncate_k and pad_zeros" begin
    x = randn(3, 4)
    res = ImageProcessing.truncate_k(x, 3, 4)
    @test res ≈ x
    res = ImageProcessing.pad_zeros(x, 3, 4)
    @test res ≈ x
    res = ImageProcessing.truncate_k(x, 1, 2)
    @test size(res) == (1, 2)
    res = ImageProcessing.pad_zeros(x, 3, 4)
    @test size(res) == (3, 4)
    x = randn(1, 2)
    res = ImageProcessing.truncate_k(x, 3, 4)
    @test res ≈ x
end

@testset "compression ratio" begin
    img = demo_image("cat.png")
    compressed = fft_compress(img, typemax(Int), typemax(Int))
    img_recovered = toimage(RGBA{N0f8}, compressed)
    @test img_recovered ≈ img
    compressed = fft_compress(img, 5, 5)
    old_size = length(img)
    new_size = 25
    @test compression_ratio(compressed) ≈ new_size / old_size
    compressed2 = lower_rank(compressed, 3, 3)
    new_size = 9
    @test compression_ratio(compressed2) ≈ new_size / old_size
    @test toimage(RGBA{N0f8}, compressed) isa AbstractArray{RGBA{N0f8}, 2}
end