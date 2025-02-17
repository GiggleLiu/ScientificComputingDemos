export AcousticPropagatorParams, solve

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
    Σx::AT
    Σy::AT
end

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
