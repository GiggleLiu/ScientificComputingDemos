using Test, LinearAlgebra, SparseArrays

@testset "fft decomposition" begin
    n = 4
    Fn = dft_matrix(n)
    F2n = dft_matrix(2n)

    # the permutation matrix to permute elements at 1:2:n (odd) to 1:n÷2 (top half)
    pm = sparse([iseven(j) ? (j÷2+n) : (j+1)÷2 for j=1:2n], 1:2n, ones(2n), 2n, 2n)

    # construct the D matrix
    ω = exp(-π*im/n)
    d1 = Diagonal([ω^(i-1) for i=1:n])

    # construct F_{2n} from F_n
    F2n_ = [Fn d1 * Fn; Fn -d1 * Fn]
    @test F2n * pm' ≈ F2n_
end