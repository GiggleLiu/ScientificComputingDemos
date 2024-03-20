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

function ifft!(x::AbstractVector{T}) where T
    N = length(x)
    @inbounds if N <= 1
        return x
    end

    # divide
    odd  = x[1:2:N]
    even = x[2:2:N]

    # conquer
    ifft!(odd)
    ifft!(even)

    # combine
    @inbounds for i=1:N÷2
       t = exp(T(2im*π*(i-1)/N)) * even[i]
       oi = odd[i]
       x[i]     = oi + t
       x[i+N÷2] = oi - t
    end
    return rmul!(x, 1/sqrt(N))
end

# here, we use the FFTW library to compute the convolution
function fast_polymul(p::AbstractVector{T}, q::AbstractVector{T}) where T
    pvals = FFTW.fft(vcat(p, zeros(T, length(q)-1)))
    qvals = FFTW.fft(vcat(q, zeros(T, length(p)-1)))
    pqvals = pvals .* qvals
    return real.(FFTW.ifft!(pqvals))
end

