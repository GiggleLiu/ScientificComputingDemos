struct Glued{T}
    data::T
end
Glued(args...) = Glued(args)

Base.zero(c::Glued) = Glued(zero.(c.data))
Base.copy(c::Glued) = Glued(copy.(c.data))
@generated function Base.zero(::Type{Glued{T}}) where T
    :(Glued($([zero(t) for t in T.types]...)))
end

@inline function Base.:(+)(a::Glued{T}, b::Glued{T}) where T
    Glued{T}(a.data .+ b.data)
end

@inline function Base.:(/)(a::Glued{T}, b::Real) where T
    Glued{T}(a.data ./ b)
end

@inline function Base.:(*)(a::Real, b::Glued{T}) where T
    Glued{T}(a .* b.data)
end

function zero_similar(arr::AbstractArray{T}, size...) where T
    zeros(T, size...)
end

function solve_detector(param::AcousticPropagatorParams, src, 
            srcv::AbstractArray{Float64, 1}, c::AbstractArray{Float64, 2}, detector_locs::AbstractVector)
    slices = zero_similar(c, length(detector_locs), param.NSTEP-1)
    tupre = zero_similar(c, param.NX+2, param.NY+2)
    tu = zero_similar(c, param.NX+2, param.NY+2)
    tφ = zero_similar(c, param.NX+2, param.NY+2)
    tψ = zero_similar(c, param.NX+2, param.NY+2)

    for i = 1:param.NSTEP-1
        tu_ = zero_similar(c, param.NX+2, param.NY+2)
        one_step!(param, tu_, tu, tupre, tφ, tψ, param.Σx, param.Σy, c)
        tu, tupre = tu_, tu
        tu[SafeIndex(src)] += srcv[i]*param.DELTAT^2
        slices[:,i] .= tu[detector_locs]
    end
    slices
end

struct GradientCache{TS,TP,TV,TP2}
    x::TS
    y::TS
    c::TP
    srcv::TV
    target_pulses::TP2
end

"""
    treeverse_solve(s0; param, src, srcv, c, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
"""
function treeverse_solve_detector(s0; param, src, srcv, c, target_pulses, detector_locs, δ=20, logger=TreeverseLog())
    f = x->treeverse_step_detector(x, param, src, srcv, c, target_pulses, detector_locs)
    res = []
    gcache = GradientCache(GVar(s0.data[2]), GVar(s0.data[2]), GVar(c), GVar(srcv), GVar(target_pulses))
    function gf(x, g)
        if g === nothing
            y = f(x)
            push!(res, y)
            g = (Glued(one(x.data[1]),zero(x.data[2])), zero(srcv), zero(c))
        end
        gy, gsrcv, gc = g
        treeverse_grad_detector(x, gy, param, src, srcv, gsrcv, c, gc, target_pulses, detector_locs, gcache)
    end
    g = treeverse(f, gf,
        copy(s0); δ=δ, N=param.NSTEP-1, f_inplace=false, logger=logger)
    res[], g
end

function treeverse_grad_detector(x_, g_, param, src, srcv, gsrcv, c, gc, target_pulses, detector_locs, gcache)
    # TODO: implement this with Enzyme.jl
end

function treeverse_step_detector(s_, param, src, srcv, c, target_pulses, detector_locs)
    l, s = s_.data
    unext, φ, ψ = zero(s.u), copy(s.φ), copy(s.ψ)
    ADSeismic.one_step!(param, unext, s.u, s.upre, φ, ψ, param.Σx, param.Σy, c)
    s2 = SeismicState(copy(s.u), unext, φ, ψ, Ref(s.step[]+1))
    s2.u[SafeIndex(src)] += srcv[s2.step[]]*param.DELTAT^2
    l += sum(abs2.(target_pulses[:,s2.step[]] .- s2.u[detector_locs]))
    return Glued(l, s2)
end