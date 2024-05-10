using ImageProcessing, ImageProcessing.Images, ImageProcessing.LinearAlgebra
using CairoMakie

fname = "amat.png"
@info "Loading image: $fname"
img = demo_image(fname)

@info "The loaded image has type: $(typeof(img))"

# the RGBA type is a 4‑tuple of red, green, blue and alpha values, each ranging from 0 to 1.
transparent = RGBA(0/255, 0/255, 0/255, 0/255)
black = RGBA(0/255, 0/255, 0/255, 255/255)
white = RGBA(255/255, 255/255, 255/255, 255/255)
red = RGBA(255/255, 0/255, 0/255, 255/255)
green = RGBA(0/255, 255/255, 0/255, 255/255)
blue = RGBA(0/255, 0/255, 255/255, 255/255)
@info """Colors are defined as:
- transparent: $transparent
- black: $black
- white: $white
- red: $red
- green: $green
- blue: $blue
"""

# get one of the color channel
red_channel = getfield.(img[:, :], :r)
# to visualize as a grayscale image
Gray.(red_channel)
fname = "red_channel.png"
Images.save(fname, Gray.(red_channel))
@info "The red channel is saved to: $fname"

# in Images, the color channels are stored as a 3D array with the first dimension being the color channel.
Gray.(channelview(img)[1, :, :])
Gray.(channelview(img)[2, :, :])
Gray.(channelview(img)[3, :, :])

red_svd = svd(red_channel)
fig, = CairoMakie.lines(red_svd.S)
fname = "red_svd_spectrum.png"
CairoMakie.save(fname, fig)
@info "Singular values of the red channel are stored in file: $fname"

# We can decompose a given image into the three color channels red, green and blue.
# Each channel can be represented as a (m × n)‑matrix with values ranging from 0 to 255.
target_rank = 10
compressed = svd_compress(img, target_rank)
compression_ratio(compressed)
newimage = toimage(RGBA{N0f8}, compressed)
fname = "compressed.png"
Images.save(fname, newimage)
@info """Compressing with SVD:
- target rank is: $target_rank
- the compression ratio is: $(compression_ratio(compressed))
- the compressed image is saved to: $fname
"""

# convert to image
toimage(RGBA{N0f8}, compressed)
compressed_rank1 = lower_rank(compressed, 1)
compression_ratio(compressed_rank1)
newimage1 = toimage(RGBA{N0f8}, compressed_rank1)
fname = "compressed_rank1.png"
Images.save(fname, newimage1)
@info """Lowering the rank to 1:
- the compression ratio is: $(compression_ratio(compressed_rank1))
- the compressed image is saved to: $fname
"""