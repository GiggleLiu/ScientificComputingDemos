# here, we use the FFTW library to compute the convolution
function fast_polymul(p::AbstractVector{T}, q::AbstractVector{T}) where T
    pvals = FFTW.fft(vcat(p, zeros(T, length(q)-1)))
    qvals = FFTW.fft(vcat(q, zeros(T, length(p)-1)))
    pqvals = pvals .* qvals
    return real.(FFTW.ifft!(pqvals))
end
