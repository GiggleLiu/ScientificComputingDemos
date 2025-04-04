module SimpleLinearAlgebra
# import packages
using LinearAlgebra
import FFTW

# `include` other source files into this module
include("strassen.jl")
include("fft.jl")
include("lu.jl")
include("qr.jl")

end # module

