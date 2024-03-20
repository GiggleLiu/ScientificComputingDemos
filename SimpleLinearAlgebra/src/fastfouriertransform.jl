#= 
We have a recursive algorithm to compute the DFT.
$F_nx=\begin{pmatrix}I_{n/2}&D_{n/2}\\I_{n/2}&-D_{n/2}\end{pmatrix}\begin{pmatrix}F_{n/2}&0\\0&F_{n/2}\end{pmatrix}\begin{pmatrix}x_{\mathrm{odd}}\\x_{\mathrm{even}}\end{pmatrix}$
where $D_n=\operatorname{diag}(1,\omega,\omega^2,\ldots,\omega^{n-1})$
We implement the O(nlogn) Cooley-Tukey algorithm to compute the DFT.
=#
function fft!(x::AbstractVector{T}) where T
    N = length(x)
    @inbounds if N <= 1
        return x
    end

    # divide
    odd  = x[1:2:N]
    even = x[2:2:N]

    # conquer
    fft!(odd)
    fft!(even)

    # combine
    @inbounds for i=1:N÷2
       t = exp(T(-2im*π*(i-1)/N)) * even[i]
       oi = odd[i]
       x[i]     = oi + t
       x[i+N÷2] = oi - t
    end
    return x
end

# A similar algorithm has already been implemented in package Polynomials. 
function fast_polymul(p::AbstractVector, q::AbstractVector)
    pvals = fft(vcat(p, zeros(length(q)-1)))
    qvals = fft(vcat(q, zeros(length(p)-1)))
    pqvals = pvals .* qvals
    return real.(ifft(pqvals))
end

