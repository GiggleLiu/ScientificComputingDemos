using LinearAlgebra

function dft_matrix(n::Int)
    ω = exp(-2π*im/n)
    return [ω^((i-1)*(j-1)) for i=1:n, j=1:n]
end
