"""
Utility functions for the ImageProcessing package.

This file contains helper functions for image loading, type conversion,
and API definitions for the compression methods.
"""

using Images

"""
    demo_image(name::String) -> AbstractArray

Load a demo image from the package data directory.

# Arguments
- `name::String`: Name of the image file. Must be one of "art.png", "cat.png", or "amat.png"

# Returns
- `AbstractArray`: The loaded image as an array of RGBA values

# Examples
```julia
julia> img = demo_image("cat.png")
julia> size(img)
(256, 256)

julia> typeof(img)
Matrix{RGBA{N0f8}}
```

# Throws
- `AssertionError`: If the image name is not valid
- `SystemError`: If the image file cannot be loaded
"""
function demo_image(name::String)
    valid_names = ["art.png", "cat.png", "amat.png"]
    if !(name in valid_names)
        throw(ArgumentError("Invalid image name '$name'. Must be one of: $(join(valid_names, ", "))"))
    end
    
    filename = pkgdir(ImageProcessing, "data", name)
    if !isfile(filename)
        throw(SystemError("Image file not found: $filename"))
    end
    
    return Images.load(filename)
end

"""
    safe_convert(::Type{N0f8}, x::T) where T -> N0f8

Safely convert a floating point number to N0f8 (8-bit fixed point).

This function clamps the input values to the valid range [0, 1] before conversion
to prevent overflow/underflow errors.

# Arguments
- `::Type{N0f8}`: Target type (8-bit fixed point)
- `x::T`: Input value(s) to convert

# Returns
- `N0f8`: Converted value(s) clamped to [0, 1]

# Examples
```julia
julia> safe_convert(N0f8, 1.5)  # Clamped to 1.0
1.0N0f8

julia> safe_convert(N0f8, -0.5)  # Clamped to 0.0
0.0N0f8
```
"""
safe_convert(::Type{N0f8}, x::T) where T = map(val -> N0f8(clamp(val, zero(T), one(T))), x)

#######################
# Abstract API Methods
#######################

"""
    lower_rank(img::Union{FFTCompressedImage, SVDCompressedImage}, args...)

Reduce the compression level of an image by further truncating coefficients.

# Methods
- `lower_rank(img::FFTCompressedImage, nx::Int, ny::Int)`: Truncate to nx×ny frequencies
- `lower_rank(img::SVDCompressedImage, rank::Int)`: Truncate to `rank` singular values

# Arguments
- `img`: Compressed image object
- For FFT: `nx::Int, ny::Int` - New frequency domain dimensions
- For SVD: `rank::Int` - New number of singular values to keep

# Returns
- Same type as input with reduced compression level

# Examples
```julia
julia> img = demo_image("cat.png")
julia> compressed = svd_compress(img, 50)
julia> further_compressed = lower_rank(compressed, 20)  # Reduce to 20 singular values
```
"""
function lower_rank end

"""
    compression_ratio(img::Union{FFTCompressedImage, SVDCompressedImage}) -> Float64

Calculate the compression ratio of a compressed image.

# Arguments
- `img`: Compressed image object (FFT or SVD compressed)

# Returns
- `Float64`: Compression ratio (compressed_size / original_size)
  - Values closer to 0 indicate better compression
  - Value of 1.0 would mean no compression

# Examples
```julia
julia> img = demo_image("cat.png")
julia> compressed = svd_compress(img, 10)
julia> ratio = compression_ratio(compressed)
0.087  # About 8.7% of original size
```
"""
function compression_ratio end

"""
    toimage(::Type{CT}, img::Union{FFTCompressedImage, SVDCompressedImage}) where CT <: Colorant

Convert a compressed image back to a standard image format.

# Arguments
- `::Type{CT}`: Target color type (e.g., `RGBA{N0f8}`)
- `img`: Compressed image object (FFT or SVD compressed)

# Returns
- `AbstractArray{CT, 2}`: Reconstructed image as a 2D array

# Examples
```julia
julia> img = demo_image("cat.png")
julia> compressed = svd_compress(img, 20)
julia> reconstructed = toimage(RGBA{N0f8}, compressed)
julia> size(reconstructed) == size(img)
true
```
"""
function toimage end