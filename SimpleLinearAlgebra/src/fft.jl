"""
    dft_matrix(n::Int) -> Matrix{Complex}

Generate the n×n Discrete Fourier Transform (DFT) matrix.

The DFT matrix W has elements W[j,k] = ω^((j-1)(k-1)) where ω = exp(-2πi/n).
This matrix transforms a vector x from time domain to frequency domain via W*x.

# Arguments
- `n::Int`: Size of the DFT matrix (must be positive)

# Returns
- `Matrix{Complex}`: The n×n DFT matrix

# Example
```julia
ulia
W = dft_matrix(4)
x = [1.0, 2.0, 3.0, 4.0]
X = W x # Compute DFT
```
"""
function dft_matrix(n::Int)
    n > 0 || throw(ArgumentError("Matrix size must be positive"))
    ω = exp(-2π*im/n)
    return [ω^((i-1)*(j-1)) for i=1:n, j=1:n]
end

"""
    fft!(x::AbstractVector{T}) where T -> AbstractVector{T}

Compute the Fast Fourier Transform (FFT) of a vector in-place using the Cooley-Tukey algorithm.

This implementation uses the radix-2 decimation-in-time FFT algorithm, which requires
the input length to be a power of 2. The algorithm has O(n log n) complexity.

The transform is defined as: X[k] = ∑(n=0...N-1) x[n] * exp(-2πi*k*n/N)

# Arguments
- `x::AbstractVector{T}`: Input vector (length must be a power of 2)

# Returns
- The input vector `x` modified in-place with its FFT

# Note
The input length must be a power of 2. For arbitrary length inputs, consider padding
with zeros to the next power of 2.
"""
function fft!(x::AbstractVector{T}) where T
    N = length(x)
    # Check if length is a power of 2
    N > 0 && (N & (N - 1) == 0) || throw(ArgumentError("Input length must be a power of 2"))
    
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
       ω = exp(T(-2im*π*(i-1)/N))  # Twiddle factor
       t = ω * even[i]
       oi = odd[i]
       x[i]     = oi + t
       x[i+N÷2] = oi - t
    end
    return x
end

"""
    ifft!(x::AbstractVector{T}) where T -> AbstractVector{T}

Compute the Inverse Fast Fourier Transform (IFFT) of a vector in-place.

This is the inverse operation of fft!. The algorithm uses the same structure as FFT
but with conjugate twiddle factors and a normalization factor of 1/√N.

The inverse transform is defined as: x[n] = (1/√N) * ∑(k=0...N-1) X[k] * exp(2πi*k*n/N)

# Arguments
- `x::AbstractVector{T}`: Input vector (length must be a power of 2)

# Returns
- The input vector `x` modified in-place with its IFFT

# Note
The input length must be a power of 2. The output is normalized by 1/√N to make
the transform unitary.
"""
function ifft!(x::AbstractVector{T}) where T
    N = length(x)
    # Check if length is a power of 2
    N > 0 && (N & (N - 1) == 0) || throw(ArgumentError("Input length must be a power of 2"))
    
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
       ω = exp(T(2im*π*(i-1)/N))  # Conjugate twiddle factor
       t = ω * even[i]
       oi = odd[i]
       x[i]     = oi + t
       x[i+N÷2] = oi - t
    end
    return rmul!(x, 1/sqrt(N))
end