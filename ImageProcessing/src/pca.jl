"""
    SVDCompressedImage{D, RT<:Real, MT<:SVD{RT}}

A compressed image representation using SVD decomposition.

# Fields
- `channels::NTuple{D, MT}`: Tuple of SVD decompositions for each color channel
  - D = 4 for RGBA images (Red, Green, Blue, Alpha channels)
  - Each channel stores U, S, V matrices from SVD decomposition

# Type Parameters
- `D`: Number of color channels (typically 4 for RGBA)
- `RT`: Real number type (Float32, Float64, etc.)
- `MT`: SVD matrix type containing the decomposition
"""
struct SVDCompressedImage{D, RT<:Real, MT<:SVD{RT}}
    channels::NTuple{D, MT}
end

"""
    svd_compress(image, k::Int) -> SVDCompressedImage

Compress an image using SVD by keeping only the k largest singular values.

The function converts the image to individual color channels, performs SVD
decomposition on each channel, and truncates to keep only the k most
significant singular values.

# Arguments
- `image`: Input image (typically RGBA format)
- `k::Int`: Number of singular values to retain (higher = better quality, less compression)

# Returns
- `SVDCompressedImage`: Compressed representation storing SVD decompositions

# Examples
```julia
julia> img = demo_image("cat.png")
julia> compressed = svd_compress(img, 20)  # Keep 20 singular values
julia> ratio = compression_ratio(compressed)
0.156  # Compressed to ~15.6% of original size
```

# See Also
- [`compression_ratio`](@ref): Calculate compression efficiency
- [`toimage`](@ref): Reconstruct image from compressed representation
"""
function svd_compress(image, k::Int)
    @assert k > 0 "Number of singular values k must be positive, got k=$k"
    
    channels = channelview(image)
    n_channels = size(channels, 1)
    
    # Apply SVD compression to each channel
    compressed_channels = ntuple(n_channels) do i
        channel_matrix = channels[i, :, :]
        truncated_svd(channel_matrix; maxrank=k)
    end
    
    return SVDCompressedImage(compressed_channels)
end

"""
    truncated_svd(m::AbstractMatrix; maxrank=typemax(Int), atol=0.0) -> SVD

Perform SVD and truncate to specified rank or tolerance.

# Arguments
- `m::AbstractMatrix`: Input matrix to decompose
- `maxrank::Int`: Maximum number of singular values to keep
- `atol::Float64`: Absolute tolerance for singular value cutoff

# Returns
- `SVD`: Truncated SVD decomposition with reduced rank
"""
function truncated_svd(m::AbstractMatrix; maxrank=typemax(Int), atol=0.0)
    full_svd = svd(m)
    return truncate(full_svd; maxrank, atol)
end

"""
    truncate(m::SVD; atol, maxrank) -> SVD

Truncate an existing SVD decomposition to specified rank or tolerance.

# Arguments
- `m::SVD`: Full SVD decomposition
- `atol::Float64`: Absolute tolerance for singular values
- `maxrank::Int`: Maximum rank to keep

# Returns
- `SVD`: Truncated SVD with reduced dimensionality
"""
function truncate(m::SVD; atol, maxrank)
    # Find cutoff index based on tolerance and maximum rank
    tol_cutoff = findlast(s -> s >= atol, m.S)
    k = min(maxrank, something(tol_cutoff, 0))
    
    if k <= 0
        throw(ArgumentError("No singular values meet the specified criteria"))
    end
    
    return SVD(m.U[:, 1:k], m.S[1:k], m.Vt[1:k, :])
end

"""
    lower_rank(img::SVDCompressedImage, k::Int) -> SVDCompressedImage

Further reduce the rank of an SVD-compressed image.

# Arguments
- `img::SVDCompressedImage`: Previously compressed image
- `k::Int`: New target rank (must be ≤ current rank)

# Returns
- `SVDCompressedImage`: Image with further reduced rank

# Examples
```julia
julia> img = demo_image("cat.png")
julia> compressed = svd_compress(img, 50)
julia> further_compressed = lower_rank(compressed, 20)
```
"""
function lower_rank(img::SVDCompressedImage, k::Int)
    if k <= 0
        throw(ArgumentError("Target rank k must be positive, got k=$k"))
    end
    
    # Check if k is valid for all channels
    for (i, channel) in enumerate(img.channels)
        if k > length(channel.S)
            throw(ArgumentError("Target rank k=$k exceeds current rank $(length(channel.S)) for channel $i"))
        end
    end
    
    # Truncate each channel to new rank
    new_channels = ntuple(length(img.channels)) do i
        truncate(img.channels[i]; atol=0, maxrank=k)
    end
    
    return SVDCompressedImage(new_channels)
end

"""
    toimage(::Type{CT}, img::SVDCompressedImage) -> AbstractArray{CT, 2}

Reconstruct an image from its SVD-compressed representation.

# Arguments
- `::Type{CT}`: Target color type (e.g., `RGBA{N0f8}`)
- `img::SVDCompressedImage`: Compressed image to reconstruct

# Returns
- `AbstractArray{CT, 2}`: Reconstructed image matrix

# Examples
```julia
julia> compressed = svd_compress(demo_image("cat.png"), 20)
julia> reconstructed = toimage(RGBA{N0f8}, compressed)
```
"""
function toimage(::Type{CT}, img::SVDCompressedImage) where {T,N,CT<:Colorant{T,N}}
    # Reconstruct each channel from SVD
    reconstructed_channels = map(img.channels) do svd_decomp
        channel_matrix = Matrix(svd_decomp)  # Reconstruct: U * Diagonal(S) * Vt
        reshape(safe_convert.(T, channel_matrix), 1, size(channel_matrix)...)
    end
    
    # Combine channels back into color image
    channel_array = cat(reconstructed_channels...; dims=1)
    return colorview(CT, channel_array)
end

"""
    compression_ratio(img::SVDCompressedImage) -> Float64

Calculate the compression ratio for an SVD-compressed image.

The ratio is computed as: (compressed_size) / (original_size)
where compressed_size accounts for storing U, S, V matrices.

# Arguments
- `img::SVDCompressedImage`: Compressed image

# Returns
- `Float64`: Compression ratio (0.0 to 1.0, lower is better compression)

# Formula
For each channel with dimensions m×n and rank k:
- Original size: m × n
- Compressed size: k × (1 + m + n)  [k singular values + k columns of U + k rows of V]
"""
function compression_ratio(img::SVDCompressedImage)
    total_compressed_size = 0
    total_original_size = 0
    
    for channel_svd in img.channels
        rank = length(channel_svd.S)
        m, n = size(channel_svd.U, 1), size(channel_svd.Vt, 2)
        
        # Original size: full m×n matrix
        original_size = m * n
        
        # Compressed size: rank*(1 + m + n) for storing S, U[:, 1:rank], Vt[1:rank, :]
        compressed_size = rank * (1 + m + n)
        
        total_compressed_size += compressed_size
        total_original_size += original_size
    end
    
    return total_compressed_size / total_original_size
end