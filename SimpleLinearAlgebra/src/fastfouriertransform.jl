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


function fast_polymul(p::AbstractVector, q::AbstractVector)
    pvals = fft(vcat(p, zeros(length(q)-1)))
    qvals = fft(vcat(q, zeros(length(p)-1)))
    pqvals = pvals .* qvals
    return real.(ifft(pqvals))
end

