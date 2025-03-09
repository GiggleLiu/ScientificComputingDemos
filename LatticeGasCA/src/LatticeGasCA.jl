module LatticeGasCA

import UnicodePlots
using CUDA.GPUArrays: @index, get_backend, @kernel
using CUDA: synchronize, CuArray
import CUDA

export cpu
export hpp_center_square, hpp_singledot, HPPLatticeGas, simulate, update!, AbstractLatticeGas, density

include("hpp.jl")
include("cuda.jl")

end
