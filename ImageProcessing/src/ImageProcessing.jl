module ImageProcessing

using LinearAlgebra
using Images
using FFTW

export demo_image, svd_compress, lower_rank, toimage, compression_ratio
export fft_compress, FFTCompressedImage
export fast_polymul

include("utils.jl")
include("fft.jl")
include("pca.jl")
include("polymul.jl")

end
