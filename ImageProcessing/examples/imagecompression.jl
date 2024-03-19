using ImageProcessing, ImageProcessing.Images, ImageProcessing.LinearAlgebra
using Makie, CairoMakie

img = demo_image("amat.png")

img isa AbstractArray
eltype(img)

# the RGBA type is a 4‑tuple of red, green, blue and alpha values, each ranging from 0 to 1.
transparent = RGBA(0/255, 0/255, 0/255, 0/255)
black = RGBA(0/255, 0/255, 0/255, 255/255)
white = RGBA(255/255, 255/255, 255/255, 255/255)
red = RGBA(255/255, 0/255, 0/255, 255/255)
green = RGBA(0/255, 255/255, 0/255, 255/255)
blue = RGBA(0/255, 0/255, 255/255, 255/255)

# get one of the color channel
red_channel = getfield.(img[:, :], :r)
# to visualize as a grayscale image
Gray.(red_channel)

# in Images, the color channels are stored as a 3D array with the first dimension being the color channel.
Gray.(channelview(img)[1, :, :])
Gray.(channelview(img)[2, :, :])
Gray.(channelview(img)[3, :, :])

red_svd = svd(red_channel)
Makie.lines(red_svd.S)

# We can decompose a given image into the three color channels red, green and blue.
# Each channel can be represented as a (m × n)‑matrix with values ranging from 0 to 255.
target_rank = 10
compressed = svd_compress(img, target_rank)
compression_ratio(compressed)

# convert to image
toimage(RGBA{N0f8}, compressed)
compressed_rank1 = lower_rank(compressed, 1)
compression_ratio(compressed_rank1)
toimage(RGBA{N0f8}, compressed_rank1)

##### FFT #####
# convert image to momentum space
img_k = fft_compress(img, size(img)...)
# the momentum space is sparse!
Gray.(real.(img_k.channels[1]) ./ sqrt(length(img)))
toimage(RGBA{N0f8}, img_k)

img_k_rank1 = lower_rank(img_k, isqrt(2 * size(img, 1)), isqrt(2 * size(img, 2)))
compression_ratio(img_k_rank1)
toimage(RGBA{N0f8}, img_k_rank1)