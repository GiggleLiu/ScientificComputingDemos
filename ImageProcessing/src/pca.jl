# We can decompose a given image into the color channels, e.g. red, green, blue and alpha.
# Each channel can be represented as a (m × n)‑matrix with values ranging from 0 to 255.
struct SVDCompressedImage{D, RT<:Real, MT<:SVD{RT}}
    channels::NTuple{D, MT}
end

"""
    svd_compress(image, k)

Convert an image to the SVD space and compress it by truncating the singular values.
`k` is the number of singular values to keep.
"""
function svd_compress(image, k::Int)
    channels = channelview(image)
    return SVDCompressedImage(ntuple(i->truncated_svd(channels[i, :, :]; maxrank=k), 4))
end
truncated_svd(m::AbstractMatrix; maxrank=typemax(Int), atol=0.0) = truncate(svd(m); maxrank, atol)
function truncate(m::SVD; atol, maxrank)
    k = min(maxrank, findlast(>=(atol), m.S))
    SVD(m.U[:, 1:k], m.S[1:k], m.Vt[1:k, :])
end
function lower_rank(img::SVDCompressedImage, k::Int)
    SVDCompressedImage(ntuple(i->truncate(img.channels[i]; atol=0, maxrank=k), 4))
end

# convert to image
function toimage(::Type{CT}, img::SVDCompressedImage) where {T,N,CT<:Colorant{T,N}}
    colorview(CT, cat([reshape(safe_convert.(T, Matrix(c)), 1, size(c)...) for c in img.channels]...; dims=1))
end

# compression ratio
function compression_ratio(img::SVDCompressedImage)
    new_size = sum(length(ch.S) * (1 + size(ch, 1) + size(ch, 2)) for ch in img.channels)
    origin = sum(prod(size(ch)) for ch in img.channels)
    return new_size / origin
end