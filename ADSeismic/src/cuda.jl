using CUDA
export togpu

function togpu(a::AcousticPropagatorParams{DIM}) where DIM
     AcousticPropagatorParams(a.NX, a.NY, a.NSTEP, a.DELTAX, a.DELTAY, a.DELTAT, CuArray(a.Σx), CuArray(a.Σy))
end

const CuSeismicState{MT} = SeismicState{MT} where MT<:CuArray

export CuSeismicState
function CuSeismicState(::Type{T}, nx::Int, ny::Int) where T
    SeismicState([CUDA.zeros(T, nx+2, ny+2) for i=1:4]..., Ref(0))
end

function togpu(x::SeismicState)
    SeismicState([CuArray(t) for t in [x.upre, x.u, x.φ, x.ψ]]..., Ref(0))
end

togpu(x::Number) = x
togpu(x::AbstractArray) = CuArray(x)

@inline function cudiv(x::Int, y::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_y = min(max_threads ÷ threads_x, y)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (x, y) ./ threads)
    threads, blocks
end

function one_step!(param::AcousticPropagatorParams, u, w, wold, φ, ψ, σ, τ, c::CuArray)
    @inline function one_step_kernel1(u, w, wold, φ, ψ, σ, τ, c, Δt, Δtx, Δty)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x + 1
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y + 1
        Δtx2 = Δtx * Δtx
        Δty2 = Δty * Δty
        Dx = 0.5Δt*Δtx
        Dy = 0.5Δt*Δty
        @inbounds if i < size(c, 1) && j < size(c, 2)
            cij = c[i,j]
            δ = (σ[i,j]+τ[i,j])*Δt*0.5
            uij = (2 - σ[i,j]*τ[i,j]*(Δt*Δt) - 2*Δtx2 * cij - 2*Δty2 * cij) * w[i,j] +
                cij * Δtx2  *  (w[i+1,j]+w[i-1,j]) +
                cij * Δty2  *  (w[i,j+1]+w[i,j-1]) +
                Dx*(φ[i+1,j]-φ[i-1,j]) +
                Dy*(ψ[i,j+1]-ψ[i,j-1]) -
                (1 - δ) * wold[i,j] 
            u[i,j] = uij / (1 + δ)
        end
        return nothing
    end

    @inline function one_step_kernel2(u, φ, ψ, σ, τ, c, Δt, Δtx_2, Δty_2)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x + 1
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y + 1
        @inbounds if i < size(c, 1) && j < size(c, 2)
            φ[i,j] = (1-Δt*σ[i,j]) * φ[i,j] + Δtx_2 * c[i,j] * (τ[i,j] -σ[i,j]) *  
                (u[i+1,j]-u[i-1,j])
            ψ[i,j] = (1-Δt*τ[i,j]) * ψ[i,j] + Δty_2 * c[i,j] * (σ[i,j] -τ[i,j]) * 
                (u[i,j+1]-u[i,j-1])
        end
        return nothing
    end

    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
 
    threads, blocks = cudiv(param.NX, param.NY)
    @cuda threads=threads blocks=blocks one_step_kernel1(u, w, wold, φ, ψ, σ, τ, c, Δt, Δt/hx, Δt/hy)
    @cuda threads=threads blocks=blocks one_step_kernel2(u, φ, ψ, σ, τ, c, Δt, 0.5*Δt/hx, 0.5*Δt/hy)
    return nothing
end


@inline function delete_state!(state::Dict{Int,<:CuSeismicState}, i::Int)
    s = pop!(state, i)
    CUDA.unsafe_free!(s.upre)
    CUDA.unsafe_free!(s.u)
    CUDA.unsafe_free!(s.φ)
    CUDA.unsafe_free!(s.ψ)
    return s
end

function Base.getindex(x::CuArray, si::SafeIndex)
    Array(x[[si.arg]])[]
end

function Base.setindex!(x::CuArray, val, si::SafeIndex)
    x[[si.arg]] = val
end