"""
    FFTCompressedImage{D, MT<:AbstractMatrix}

A compressed image representation using FFT in frequency domain.

# Fields
- `Nx::Int`: Original image width
- `Ny::Int`: Original image height  
- `channels::NTuple{D, MT}`: Tuple of frequency domain matrices for each color channel
  - D = 4 for RGBA images (Red, Green, Blue, Alpha channels)
  - Each channel contains truncated frequency coefficients

# Type Parameters
- `D`: Number of color channels (typically 4 for RGBA)
- `MT`: Matrix type for storing frequency coefficients (typically Complex{Float64})
"""
struct FFTCompressedImage{D, MT<:AbstractMatrix}
    Nx::Int
    Ny::Int
    channels::NTuple{D, MT}
    
    # Inner constructor with validation
    function FFTCompressedImage{D, MT}(Nx::Int, Ny::Int, channels::NTuple{D, MT}) where {D, MT<:AbstractMatrix}
        if Nx <= 0 || Ny <= 0
            throw(ArgumentError("Image dimensions must be positive: Nx=$Nx, Ny=$Ny"))
        end
        if length(channels) != D
            throw(ArgumentError("Expected $D channels, got $(length(channels))"))
        end
        new{D, MT}(Nx, Ny, channels)
    end
end

# Convenience constructor
FFTCompressedImage(Nx::Int, Ny::Int, channels::NTuple{D, MT}) where {D, MT} = 
    FFTCompressedImage{D, MT}(Nx, Ny, channels)

"""
    fft_compress(image, nx::Int, ny::Int) -> FFTCompressedImage

Compress an image using FFT by keeping only specified frequency components.

The function transforms each color channel to frequency domain using FFT,
shifts zero frequency to center, and truncates to keep only nx×ny frequency
components around the center (low frequencies).

# Arguments
- `image`: Input image (typically RGBA format)
- `nx::Int`: Number of frequency components to keep in x-direction
- `ny::Int`: Number of frequency components to keep in y-direction  

# Returns
- `FFTCompressedImage`: Compressed representation in frequency domain

# Examples
```julia
julia> img = demo_image("cat.png")
julia> compressed = fft_compress(img, 64, 64)  # Keep 64×64 frequencies
julia> ratio = compression_ratio(compressed)
0.25  # Compressed to 25% of original size
```

# Notes
- Lower frequencies (near center) contain most image information
- Higher nx, ny values preserve more detail but reduce compression
"""
function fft_compress(image, nx::Int, ny::Int)
    if nx <= 0 || ny <= 0
        throw(ArgumentError("Frequency dimensions must be positive: nx=$nx, ny=$ny"))
    end
    
    channels = channelview(image)
    original_size = size(image)
    n_channels = size(channels, 1)
    
    # Transform each channel to frequency domain and truncate
    compressed_channels = ntuple(n_channels) do i
        channel_matrix = channels[i, :, :]
        
        # FFT -> shift zero frequency to center -> truncate
        freq_domain = fftshift(fft(channel_matrix))
        truncate_k(freq_domain, nx, ny)
    end
    
    return FFTCompressedImage(original_size..., compressed_channels)
end

"""
    truncated_fft(m::AbstractMatrix, nx::Int, ny::Int) -> AbstractMatrix

Transform matrix to frequency domain and truncate (alternative interface).

# Arguments
- `m::AbstractMatrix`: Input matrix
- `nx::Int, ny::Int`: Target frequency domain dimensions

# Returns
- `AbstractMatrix`: Truncated frequency domain representation
"""
truncated_fft(m::AbstractMatrix, nx::Int, ny::Int) = truncate_k(fftshift(fft(m)), nx, ny)

"""
    truncate_k(m::AbstractMatrix, nx::Int, ny::Int) -> AbstractMatrix

Truncate a frequency domain matrix to specified dimensions around center.

Extracts nx×ny coefficients centered around the zero frequency component.
This preserves the most important low-frequency information.

# Arguments
- `m::AbstractMatrix`: Input frequency domain matrix
- `nx::Int`: Target width (≤ original width)
- `ny::Int`: Target height (≤ original height)

# Returns
- `AbstractMatrix`: Truncated matrix of size nx×ny

# Examples
```julia
julia> freq_matrix = fftshift(fft(randn(100, 100)))
julia> truncated = truncate_k(freq_matrix, 20, 20)  # Keep center 20×20
julia> size(truncated)
(20, 20)
```
"""
function truncate_k(m::AbstractMatrix, nx::Int, ny::Int)
    if nx <= 0 || ny <= 0
        throw(ArgumentError("Truncation dimensions must be positive: nx=$nx, ny=$ny"))
    end
    
    original_nx, original_ny = size(m)
    
    # Ensure we don't truncate to larger than original
    nx = min(nx, original_nx)
    ny = min(ny, original_ny)
    
    # Calculate center position and extraction window
    center_x = (original_nx + 1) ÷ 2
    center_y = (original_ny + 1) ÷ 2
    
    # Calculate start indices for extraction
    start_x = center_x - (nx - 1) ÷ 2
    start_y = center_y - (ny - 1) ÷ 2
    
    # Extract the centered submatrix
    return m[start_x:(start_x + nx - 1), start_y:(start_y + ny - 1)]
end

"""
    pad_zeros(m::AbstractMatrix{T}, Nx::Int, Ny::Int) -> AbstractMatrix{T}

Pad a matrix with zeros to specified dimensions, centering the original content.

This is used during reconstruction to restore the original frequency domain size
before inverse FFT.

# Arguments
- `m::AbstractMatrix{T}`: Input matrix to pad
- `Nx::Int`: Target width  
- `Ny::Int`: Target height

# Returns
- `AbstractMatrix{T}`: Zero-padded matrix of size Nx×Ny

# Examples
```julia
julia> small_matrix = randn(ComplexF64, 20, 20)
julia> padded = pad_zeros(small_matrix, 100, 100)
julia> size(padded)
(100, 100)
```
"""
function pad_zeros(m::AbstractMatrix{T}, Nx::Int, Ny::Int) where T
    if Nx <= 0 || Ny <= 0
        throw(ArgumentError("Padding dimensions must be positive: Nx=$Nx, Ny=$Ny"))
    end
    
    original_nx, original_ny = size(m)
    
    # Create output matrix filled with zeros
    output = zeros(T, Nx, Ny)
    
    # Calculate center position for placing the original matrix
    center_x = (Nx + 1) ÷ 2
    center_y = (Ny + 1) ÷ 2
    
    # Calculate start indices for placement
    start_x = center_x - (original_nx - 1) ÷ 2
    start_y = center_y - (original_ny - 1) ÷ 2
    
    # Place original matrix in center
    end_x = start_x + original_nx - 1
    end_y = start_y + original_ny - 1
    
    output[start_x:end_x, start_y:end_y] = m
    
    return output
end

"""
    lower_rank(img::FFTCompressedImage, nx::Int, ny::Int) -> FFTCompressedImage

Further reduce the frequency content of an FFT-compressed image.

# Arguments
- `img::FFTCompressedImage`: Previously compressed image
- `nx::Int`: New frequency width (must be ≤ current width)
- `ny::Int`: New frequency height (must be ≤ current height)

# Returns
- `FFTCompressedImage`: Image with further reduced frequency content
"""
function lower_rank(img::FFTCompressedImage, nx::Int, ny::Int)
    if nx <= 0 || ny <= 0
        throw(ArgumentError("Target dimensions must be positive: nx=$nx, ny=$ny"))
    end
    
    # Verify new dimensions don't exceed current ones
    for (i, channel) in enumerate(img.channels)
        curr_nx, curr_ny = size(channel)
        if nx > curr_nx || ny > curr_ny
            throw(ArgumentError("Target dimensions ($nx×$ny) exceed current dimensions ($(curr_nx)×$(curr_ny)) for channel $i"))
        end
    end
    
    # Truncate each channel further
    new_channels = ntuple(length(img.channels)) do i
        truncate_k(img.channels[i], nx, ny)
    end
    
    return FFTCompressedImage(img.Nx, img.Ny, new_channels)
end

"""
    toimage(::Type{CT}, img::FFTCompressedImage) -> AbstractArray{CT, 2}

Reconstruct an image from its FFT-compressed representation.

The reconstruction process:
1. Pad frequency coefficients back to original size
2. Shift zero frequency back to corner
3. Apply inverse FFT to get spatial domain
4. Take real part and convert to target color type

# Arguments
- `::Type{CT}`: Target color type (e.g., `RGBA{N0f8}`)
- `img::FFTCompressedImage`: Compressed image to reconstruct

# Returns
- `AbstractArray{CT, 2}`: Reconstructed image matrix
"""
function toimage(::Type{CT}, img::FFTCompressedImage) where {T,N,CT<:Colorant{T,N}}
    # Reconstruct each channel from frequency domain
    reconstructed_channels = map(img.channels) do channel_freq
        # Pad back to original size
        padded_freq = pad_zeros(channel_freq, img.Nx, img.Ny)
        
        # Shift zero frequency back to corner and apply inverse FFT
        spatial_domain = ifft(ifftshift(padded_freq))
        
        # Take real part (imaginary should be ~0 for real images)
        real_spatial = real.(spatial_domain)
        
        # Convert to target type and reshape for colorview
        converted = safe_convert.(T, real_spatial)
        reshape(converted, 1, img.Nx, img.Ny)
    end
    
    # Combine channels back into color image
    channel_array = cat(reconstructed_channels...; dims=1)
    return colorview(CT, channel_array)
end

"""
    compression_ratio(img::FFTCompressedImage) -> Float64

Calculate the compression ratio for an FFT-compressed image.

# Arguments
- `img::FFTCompressedImage`: Compressed image

# Returns
- `Float64`: Compression ratio (compressed_size / original_size)
"""
function compression_ratio(img::FFTCompressedImage)
    # Original size: Nx × Ny × number_of_channels
    original_size = img.Nx * img.Ny * length(img.channels)
    
    # Compressed size: sum of all frequency coefficients stored
    compressed_size = sum(length(channel) for channel in img.channels)
    
    return compressed_size / original_size
end