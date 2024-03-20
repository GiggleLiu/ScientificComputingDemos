using ImageProcessing, ImageProcessing.Images, ImageProcessing.LinearAlgebra

fname = "amat.png"
@info "Running FFT compression example, loaded image: $fname"
img = demo_image(fname)

##### FFT #####
img_k = fft_compress(img, size(img)...)
# the momentum space is sparse!
red_channel = Gray.(real.(img_k.channels[1]) ./ sqrt(length(img)))
fname = "red-momentum_space.png"
Images.save(fname, red_channel)
@info "Converting image to momentum space, red channel saved to: $fname"
Images.save(fname, toimage(RGBA{N0f8}, img_k))
fname = "recovered.png"
@info "Recovered image from momentum space is saved to: $fname"

nx, ny = isqrt(2 * size(img, 1)), isqrt(2 * size(img, 2))
img_k_fft = lower_rank(img_k, nx, ny)
cratio = compression_ratio(img_k_fft)
fname = "fft_compressed.png"
Images.save(fname, toimage(RGBA{N0f8}, img_k_fft))
@info "Compressing to size: $nx x $ny, compression ratio: $cratio, saved to: $fname"