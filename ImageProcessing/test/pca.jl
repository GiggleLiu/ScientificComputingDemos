using Test, ImageProcessing, ImageProcessing.Images

@testset "compression ratio" begin
    img = demo_image("cat.png")
    compressed = svd_compress(img, 5)
    old_size = length(img)
    new_size = size(img, 1) * 5 + 5 + size(img, 2) * 5
    @test compression_ratio(compressed) ≈ new_size / old_size
    compressed2 = lower_rank(compressed, 3)
    new_size = size(img, 1) * 3 + 3 + size(img, 2) * 3
    @test compression_ratio(compressed2) ≈ new_size / old_size
    @test toimage(RGBA{N0f8}, compressed) isa AbstractArray{RGBA{N0f8}, 2}
end

@testset "truncated svd" begin
    x = randn(10, 3)
    A = x * x'
    res = ImageProcessing.truncated_svd(A; maxrank=3, atol=0)
    @test Matrix(res) ≈ A
end