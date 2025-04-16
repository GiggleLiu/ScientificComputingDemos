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

function zero_similar(arr::AbstractArray{T}, size...) where T
    res = similar(arr, size...)
    fill!(res, zero(T))
    return res
end

function treeverse_grad_detector(lx, x, lg, g, param, src, srcv, gsrcv, c, gc, target_pulses, detector_locs, gcache)
    @info "gradient: $(x.step[]+1) -> $(x.step[])"
    ly, y = treeverse_step_detector(x_, param, src, srcv, c, target_pulses, detector_locs).data

    # fit data into cache
    gcache.c .= GVar.(c, gc)
    gcache.srcv .= GVar.(srcv, gsrcv)
    for field in fieldnames(SeismicState)[1:end-1]
        getfield(gcache.y, field) .= GVar.(getfield(y, field), getfield(g, field))
        getfield(gcache.x, field) .= GVar.(getfield(x, field))
    end
    gcache.x.step[] = x.step[]
    gcache.y.step[] = y.step[]

    # compute
    _, gs, _, _, gv, gc2 = (~bennett_step_detector!)(Glued(GVar(ly, lg), gcache.y), Glued(GVar(lx), gcache.x), param, src, gcache.srcv, gcache.c, gcache.target_pulses, detector_locs)

    # get gradients from the cache
    gc .= grad(gc2)
    gsrcv .= grad.(gv)
    for field in fieldnames(SeismicState)[1:end-1]
        getfield(g, field) .= grad.(getfield(gs.data[2], field))
    end

    return (Glued(grad(gs.data[1]), g), gsrcv, gc)
end

function treeverse_step_detector(s_, param, src, srcv, c, target_pulses, detector_locs)
    l, s = s_.data
    unext, φ, ψ = zero(s.u), copy(s.φ), copy(s.ψ)
    ReversibleSeismic.one_step!(param, unext, s.u, s.upre, φ, ψ, param.Σx, param.Σy, c)
    s2 = SeismicState(copy(s.u), unext, φ, ψ, Ref(s.step[]+1))
    s2.u[SafeIndex(src)] += srcv[s2.step[]]*param.DELTAT^2
    l += sum(abs2.(target_pulses[:,s2.step[]] .- s2.u[detector_locs]))
    return (l, s2)
end

function treeverse_solve_detector(ls, s0; param, src, srcv, c, target_pulses, detector_locs, δ=20, logger=TreeverseLog())
    f = x->treeverse_step_detector(x, param, src, srcv, c, target_pulses, detector_locs)
    res = []
    gcache = (copy(s0), copy(s0), copy(c), copy(srcv), GVar(target_pulses))
    function gf(x, g)
        if g === nothing
            y = f(x)
            push!(res, y)
            g = ((one(x[1]),zero(x[2])), zero(srcv), zero(c))
        end
        gy, gsrcv, gc = g
        treeverse_grad_detector(lx, x, ly, gy, param, src, srcv, gsrcv, c, gc, target_pulses, detector_locs, gcache)
    end
    g = treeverse(f, gf,
        copy(s0); δ=δ, N=param.NSTEP-1, f_inplace=false, logger=logger)
    res[], g
end
