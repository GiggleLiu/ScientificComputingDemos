"""
# ImageProcessing

A Julia package for demonstrating image compression techniques using SVD and FFT.

This package provides educational implementations of:
- SVD (Singular Value Decomposition) based image compression  
- FFT (Fast Fourier Transform) based image compression
- Utilities for loading demo images and comparing compression methods

## Example Usage

```julia
using ImageProcessing

# Load a demo image
img = demo_image("cat.png")

# SVD compression
compressed_svd = svd_compress(img, 20)  # Keep 20 singular values
ratio_svd = compression_ratio(compressed_svd)
decompressed_svd = toimage(RGBA{N0f8}, compressed_svd)

# FFT compression  
compressed_fft = fft_compress(img, 64, 64)  # Keep 64×64 frequencies
ratio_fft = compression_ratio(compressed_fft)
decompressed_fft = toimage(RGBA{N0f8}, compressed_fft)
```
"""
module ImageProcessing

using LinearAlgebra
using Images
using FFTW

# Export main functions
export demo_image
export svd_compress, fft_compress
export lower_rank, compression_ratio, toimage

# Export image types
export SVDCompressedImage, FFTCompressedImage

# Include implementation files
include("utils.jl")
include("pca.jl")
include("fft.jl")

end