#= 
It defines a function dft_matrix that generates the matrix for the Discrete Fourier Transform (DFT) of size n. 
The DFT is a mathematical technique used in signal processing to transform a sequence of complex numbers from the time domain to the frequency domain.
=#
using LinearAlgebra
function dft_matrix(n::Int)
    ω = exp(-2π*im/n)
    return [ω^((i-1)*(j-1)) for i=1:n, j=1:n]
end
