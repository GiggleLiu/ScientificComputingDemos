"""
    demo_image(name::String)

Load an image from the ImageProcessing package data directory.
The argument `name` can be "art.png", "cat.png" or "amat.png"
"""
function demo_image(name::String)
    @assert name in ["art.png", "cat.png", "amat.png"] "Invalid image name, should be one of 'art.png', 'cat.png' or 'amat.png', got : $name."
    filename = pkgdir(ImageProcessing, "data", name)
    return Images.load(filename)
end

# convert floating point number to N0f8, fixed point number with 8 bits, safely
safe_convert(::Type{N0f8}, x::T) where T = map(x->N0f8(min(max(x, zero(T)), one(T))), x)

##### APIs #####
"""
    lower_rank(img::FFTCompressedImage, nx::Int, ny::Int)
    lower_rank(img::SVDCompressedImage, rank::Int)

Lower the size of the image by truncating the Fourier coefficients or the singular values.
"""
function lower_rank end

"""
    compression_ratio(img::FFTCompressedImage)
    compression_ratio(img::SVDCompressedImage)

Compute the compression ratio of the compressed image.
"""
function compression_ratio end

"""
    toimage(::Type{CT}, img::FFTCompressedImage)
    toimage(::Type{CT}, img::SVDCompressedImage)

Convert a compressed image to an image, an array of elements of type `CT`.
"""
function toimage end