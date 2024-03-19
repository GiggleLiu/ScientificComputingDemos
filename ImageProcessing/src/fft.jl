struct FFTCompressedImage{D, MT<:AbstractMatrix}
    Nx::Int
    Ny::Int
    channels::NTuple{D, MT}
end

"""
    fft_compress(image, nx, ny)

Convert an image to momentum space using the FFT algorithm and compress it by truncating the Fourier coefficients.
`nx` and `ny` are the number of rows and columns to keep in the Fourier space.
"""
function fft_compress(image, nx::Int, ny::Int)
    channels = channelview(image)
    return FFTCompressedImage(size(image)..., ntuple(i->truncate_k(channels[i, :, :] |> fft |> fftshift, nx, ny), 4))
end
truncated_fft(m::AbstractMatrix, nx::Int, ny::Int) = truncate_k(fftshift(fft(m)), nx, ny)
function truncate_k(m::AbstractMatrix, nx::Int, ny::Int)
    nx = min(nx, size(m, 1))
    ny = min(ny, size(m, 2))
    startx = (size(m, 1) + 1) ÷ 2 - (nx-1) ÷ 2
    starty = (size(m, 2) + 1) ÷ 2 - (ny-1) ÷ 2
    return m[startx:startx+nx-1, starty:starty+ny-1]
end

# pad zeros to matrix m, to make it of size Nx x Ny
function pad_zeros(m::AbstractMatrix{T}, Nx::Int, Ny::Int) where T
    output = similar(m, Nx, Ny)
    fill!(output, zero(T))
    Nx = max(Nx, size(m, 1))
    Ny = max(Ny, size(m, 2))
    startx = (Nx + 1) ÷ 2 - (size(m, 1)-1) ÷ 2
    starty = (Ny + 1) ÷ 2 - (size(m, 2)-1) ÷ 2
    output[startx:startx+size(m, 1)-1, starty:starty+size(m, 2)-1] .= m
    return output
end

# lower the size of the image by truncating the Fourier coefficients
function lower_rank(img::FFTCompressedImage, nx::Int, ny::Int)
    FFTCompressedImage(img.Nx, img.Ny, ntuple(i->truncate_k(img.channels[i], nx, ny), 4))
end

# convert to image
function toimage(::Type{CT}, img::FFTCompressedImage) where {T,N,CT<:Colorant{T,N}}
    colorview(CT, cat([reshape(safe_convert.(T, pad_zeros(c, img.Nx, img.Ny) |> ifftshift |> ifft! .|> real), 1, img.Nx, img.Ny) for c in img.channels]...; dims=1))
end

# compression ratio
function compression_ratio(img::FFTCompressedImage)
    new_size = sum(length(ch) for ch in img.channels)
    return new_size / (img.Nx * img.Ny * length(img.channels))
end