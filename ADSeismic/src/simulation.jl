# This file implements the core structures and algorithms for acoustic wave propagation simulation
# with Perfectly Matched Layer (PML) boundary conditions.

struct SeismicState{MT}
    upre::MT    # Previous wavefield
    u::MT       # Current wavefield
    φ::MT       # PML auxiliary variable for x-direction
    ψ::MT       # PML auxiliary variable for y-direction
    step::Base.RefValue{Int}  # Current time step
end
## solving gradient
function SeismicState(::Type{T}, nx::Int, ny::Int) where T
    SeismicState([zeros(T, nx+2, ny+2) for i=1:4]..., Ref(0))
end
Base.zero(x::SeismicState) = SeismicState(zero(x.upre), zero(x.u), zero(x.φ), zero(x.ψ), Ref(0))
Base.copy(x::SeismicState) = SeismicState(copy(x.upre), copy(x.u), copy(x.φ), copy(x.ψ), Ref(x.step[]))

struct AcousticPropagatorParams{DIM, AT<:AbstractArray{Float64,DIM}}
    # number of grids along x,y axis and time steps
    NX::Int
    NY::Int 
    NSTEP::Int

    # size of grid cell and time step
    DELTAX::Float64
    DELTAY::Float64
    DELTAT::Float64

    # Auxilliary Data
    Σx::AT    # PML damping coefficients in x-direction
    Σy::AT    # PML damping coefficients in y-direction
end

# Constructor for AcousticPropagatorParams with PML configuration
function AcousticPropagatorParams(; nx::Int, ny::Int, nstep::Int,
        dx::Float64, dy::Float64, dt::Float64,
        Rcoef::Float64=0.001, # Relative reflection coefficient
        vp_ref::Float64=1000.0,
        npoints_PML::Int=12,
        USE_PML_XMAX::Bool = true,
        USE_PML_XMIN::Bool = true,
        USE_PML_YMAX::Bool = true,
        USE_PML_YMIN::Bool = true)
    # computing damping coefficient
    Lx = npoints_PML * dx
    Ly = npoints_PML * dy
    damping_x = vp_ref/Lx*log(1/Rcoef)
    damping_y = vp_ref/Ly*log(1/Rcoef)

    Σx, Σy = zeros(nx+2, ny+2), zeros(nx+2, ny+2)
    for i = 1:nx+2
        for j = 1:ny+2
            Σx[i,j] = pml_helper((i-1)*dx, nx, dx,
                damping_x, npoints_PML,
                USE_PML_XMIN, USE_PML_XMAX)
            Σy[i,j] = pml_helper((j-1)*dy, ny, dy,
                damping_y, npoints_PML,
                USE_PML_YMIN, USE_PML_YMAX)
        end
    end
    return AcousticPropagatorParams(nx, ny, nstep, dx, dy, dt, Σx, Σy)
end

# Helper function to calculate PML damping profile
function pml_helper(x::Float64, nx::Int, dx::Float64, ξx::Float64, npoints_PML::Int,
        USE_PML_XMIN, USE_PML_XMAX)
    Lx = npoints_PML * dx
    out = 0.0
    if x<Lx && USE_PML_XMIN 
        d = abs(Lx-x)
        out = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    elseif x>dx*(nx+1)-Lx && USE_PML_XMAX
        d = abs(x-(dx*(nx+1)-Lx))
        out = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    end
    return out
end

# Performs a single time step update of the acoustic wave equation with PML
function one_step!(param::AcousticPropagatorParams, u, w, wold, φ, ψ, σ, τ, c)
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
 
    @inbounds for j=2:param.NY+1, i=2:param.NX+1
        uij = (2 - σ[i,j]*τ[i,j]*Δt^2 - 2*Δt^2/hx^2 * c[i,j] - 2*Δt^2/hy^2 * c[i,j]) * w[i,j] +
            c[i,j] * (Δt/hx)^2  *  (w[i+1,j]+w[i-1,j]) +
            c[i,j] * (Δt/hy)^2  *  (w[i,j+1]+w[i,j-1]) +
            (Δt^2/(2hx))*(φ[i+1,j]-φ[i-1,j]) +
            (Δt^2/(2hy))*(ψ[i,j+1]-ψ[i,j-1]) -
            (1 - (σ[i,j]+τ[i,j])*Δt/2) * wold[i,j] 
        u[i,j] = uij / (1 + (σ[i,j]+τ[i,j])/2*Δt)
    end
    @inbounds for j=2:param.NY+1, i=2:param.NX+1
        φ[i,j] = (1. -Δt*σ[i,j]) * φ[i,j] + Δt * c[i,j] * (τ[i,j] -σ[i,j])/2hx *  
            (u[i+1,j]-u[i-1,j])
        ψ[i,j] = (1-Δt*τ[i,j]) * ψ[i,j] + Δt * c[i,j] * (σ[i,j] -τ[i,j])/2hy * 
            (u[i,j+1]-u[i,j-1])
    end
end

# Solves the acoustic wave equation and returns the full wavefield history
function solve(param::AcousticPropagatorParams, src, 
            srcv::Array{Float64, 1}, c::Array{Float64, 2})

    tu = zeros(param.NX+2, param.NY+2, param.NSTEP+1)
    tφ = zeros(param.NX+2, param.NY+2)
    tψ = zeros(param.NX+2, param.NY+2)

    for i = 3:param.NSTEP+1
        one_step!(param, view(tu,:,:,i), view(tu,:,:,i-1), view(tu,:,:,i-2), tφ, tψ, param.Σx, param.Σy, c)
        tu[src[1], src[2], i] += srcv[i-2]*param.DELTAT^2
    end
    tu
end

# Solves the acoustic wave equation and returns only the final wavefield (memory efficient)
function solve_final(param::AcousticPropagatorParams, src, 
            srcv::Array{Float64, 1}, c::Array{Float64, 2})

    tupre = zeros(param.NX+2, param.NY+2)
    tu = zeros(param.NX+2, param.NY+2)
    tφ = zeros(param.NX+2, param.NY+2)
    tψ = zeros(param.NX+2, param.NY+2)

    for i = 3:param.NSTEP+1
        tu_ = zeros(param.NX+2, param.NY+2)
        one_step!(param, tu_, tu, tupre, tφ, tψ, param.Σx, param.Σy, c)
        tu, tupre = tu_, tu
        tu[src...] += srcv[i-2]*param.DELTAT^2
    end
    tu
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

