using ImageProcessing, ImageProcessing.Images, ImageProcessing.LinearAlgebra
using CairoMakie
using CairoMakie: Axis

"""
Image Compression Demo using SVD and FFT
========================================

This demo shows how to compress images using two different methods:
1. SVD (Singular Value Decomposition) - keeps the most important "patterns"
2. FFT (Fast Fourier Transform) - keeps the most important frequencies
"""

function run_compression_demo()
    println("🖼️  Image Compression Demonstration")
    println("="^50)
    
    # Load a demo image
    img_name = "cat.png"
    @info "Loading image: $img_name"
    img = demo_image(img_name)
    @info "Image size: $(size(img))"
    
    # Save original image
    Images.save("original.png", img)
    @info "Original image saved as: original.png"
    
    # Demonstrate SVD compression
    demonstrate_svd_compression(img)
    
    # Demonstrate FFT compression
    demonstrate_fft_compression(img)
    
    # Create comparison plot
    create_comparison_plot(img)
    
    println("\n🎉 Demo complete! Check the generated images to see compression effects.")
end

"""
Demonstrate SVD compression with different ranks
"""
function demonstrate_svd_compression(img)
    println("\n📊 SVD Compression:")
    println("-"^20)
    
    # Test different compression levels (number of singular values to keep)
    ranks = [1, 5, 10, 20, 50]
    
    for rank in ranks
        # Compress the image
        compressed = svd_compress(img, rank)
        ratio = compression_ratio(compressed)
        compressed_img = toimage(RGBA{N0f8}, compressed)
        
        # Save compressed image
        filename = joinpath(@__DIR__, "svd_rank$(rank)_ratio$(round(ratio, digits=3)).png")
        Images.save(filename, compressed_img)
        
        @info "Rank $rank: compression ratio = $(round(ratio, digits=3)), saved as $filename"
    end
end

"""
Demonstrate FFT compression with different frequency cutoffs
"""
function demonstrate_fft_compression(img)
    println("\n🌊 FFT Compression:")
    println("-"^20)
    
    original_size = size(img)
    
    # Convert to frequency domain
    img_k = fft_compress(img, original_size...)
    
    # Save frequency domain visualization (red channel)
    red_freq = abs.(img_k.channels[1])
    red_freq_normalized = red_freq ./ maximum(red_freq)
    Images.save("frequency_domain.png", Gray.(red_freq_normalized))
    @info "Frequency domain visualization saved as: frequency_domain.png"
    
    # Test different compression levels (keep different amounts of frequencies)
    compression_factors = [2, 4, 6, 8]
    
    for factor in compression_factors
        # Calculate new size (smaller = more compression)
        nx = Int(round(original_size[1] / factor))
        ny = Int(round(original_size[2] / factor))
        
        # Compress in frequency domain
        compressed_fft = lower_rank(img_k, nx, ny)
        ratio = compression_ratio(compressed_fft)
        compressed_img = toimage(RGBA{N0f8}, compressed_fft)
        
        # Save compressed image
        filename = joinpath(@__DIR__, "fft_$(nx)x$(ny)_ratio$(round(ratio, digits=3)).png")
        Images.save(filename, compressed_img)
        
        @info "FFT $(nx)×$(ny): compression ratio = $(round(ratio, digits=3)), saved as $filename"
    end
end

"""
Create a comparison plot showing different compression methods
"""
function create_comparison_plot(img)
    println("\n📈 Creating comparison plot...")
    
    # Create a figure showing original vs compressed versions
    fig = Figure(size=(1200, 800))
    
    # Original image
    ax1 = Axis(fig[1, 1], title="Original", aspect=DataAspect())
    image!(ax1, rotr90(img))
    hidespines!(ax1); hidedecorations!(ax1)
    
    # SVD compressed (rank 10)
    svd_compressed = svd_compress(img, 10)
    svd_img = toimage(RGBA{N0f8}, svd_compressed)
    svd_ratio = compression_ratio(svd_compressed)
    
    ax2 = Axis(fig[1, 2], title="SVD (rank 10)\nRatio: $(round(svd_ratio, digits=3))", aspect=DataAspect())
    image!(ax2, rotr90(svd_img))
    hidespines!(ax2); hidedecorations!(ax2)
    
    # FFT compressed
    original_size = size(img)
    img_k = fft_compress(img, original_size...)
    nx, ny = original_size[1] ÷ 6, original_size[2] ÷ 6
    fft_compressed = lower_rank(img_k, nx, ny)
    fft_img = toimage(RGBA{N0f8}, fft_compressed)
    fft_ratio = compression_ratio(fft_compressed)
    
    ax3 = Axis(fig[1, 3], title="FFT ($(nx)×$(ny))\nRatio: $(round(fft_ratio, digits=3))", aspect=DataAspect())
    image!(ax3, rotr90(fft_img))
    hidespines!(ax3); hidedecorations!(ax3)
    
    # Show singular values plot
    red_channel = getfield.(img[:, :], :r)
    red_svd = svd(red_channel)
    
    ax4 = Axis(fig[2, 1:3], 
        xlabel="Index", 
        ylabel="Singular Value",
        title="SVD Spectrum (Red Channel)",
        yscale=log10)
    lines!(ax4, 1:min(100, length(red_svd.S)), red_svd.S[1:min(100, length(red_svd.S))], 
           color=:red, linewidth=2)

    filename = joinpath(@__DIR__, "compression_comparison.png")
    save(filename, fig)
    @info "Comparison plot saved as: $filename"
end

# Run the demo
run_compression_demo()