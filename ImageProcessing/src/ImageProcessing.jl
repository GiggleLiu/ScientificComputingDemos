module ImageProcessing

using LinearAlgebra
using Images
using FFTW

export demo_image, svd_compress, lower_rank, toimage, compression_ratio
export fft_compress, FFTCompressedImage

include("utils.jl")
include("fft.jl")
include("pca.jl")

end
