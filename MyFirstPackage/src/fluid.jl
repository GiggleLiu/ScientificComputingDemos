# Reference: https://physics.weber.edu/schroeder/fluids/
"""
    AbstractLBConfig{D, N}

An abstract type for lattice Boltzmann configurations.
"""
abstract type AbstractLBConfig{D, N} end
    
"""
    D2Q9 <: AbstractLBConfig{2, 9}

A lattice Boltzmann configuration for 2D, 9-velocity model.
"""
struct D2Q9 <: AbstractLBConfig{2, 9} end
directions(::D2Q9) = (
        Point(1, 1),
        Point(-1, 1),
        Point(1, 0),
        Point(0, -1),
        Point(0, 0),
        Point(0, 1),
        Point(-1, 0),
        Point(1, -1),
        Point(-1, -1),
    )

# directions[k] is the opposite of directions[flip_direction_index(k)
function flip_direction_index(::D2Q9, i::Int)
    return 10 - i
end
# the distribution of the 9 velocities at the equilibrium state
weights(::D2Q9) = (1/36, 1/36, 1/9, 1/9, 4/9, 1/9, 1/9, 1/36, 1/36)

# the density of the fluid, each component is the density of a velocity
struct Cell{N, T <: Real}
    density::NTuple{N, T}
end
Base.isapprox(x::Cell, y::Cell; kwargs...) = all(isapprox.(x.density, y.density; kwargs...))
# the total desnity of the fluid
density(cell::Cell) = sum(cell.density)
# the density of the fluid in a specific direction,
# where the direction is an integer
density(cell::Cell, direction::Int) = cell.density[direction]

"""
    momentum(lb::AbstractLBConfig, rho::Cell)

Compute the momentum of the fluid from the density of the fluid.
"""
function momentum(lb::AbstractLBConfig, rho::Cell)
    return mapreduce((r, d) -> r * d, +, rho.density, directions(lb)) / density(rho)
end
Base.:+(x::Cell, y::Cell) = Cell(x.density .+ y.density)
Base.:*(x::Real, y::Cell) = Cell(x .* y.density)

"""
    equilibrium_density(lb::AbstractLBConfig, ρ, u)

Compute the equilibrium density of the fluid from the total density and the momentum.
"""
function equilibrium_density(lb::AbstractLBConfig{<:Any, N}, ρ, u) where {N}
    ws, ds = weights(lb), directions(lb)
    return Cell(
        ntuple(i-> ρ * ws[i] * _equilibrium_density(u, ds[i]), N)
    )
end
function _equilibrium_density(u, ei)
    # the equilibrium density of the fluid with a specific momentum
    return (1 + 3 * dot(ei, u) + 9/2 * dot(ei, u)^2 - 3/2 * dot(u, u))
end

# streaming step
function stream!(lb::AbstractLBConfig{2, N},
        newgrid::AbstractMatrix{D},
        grid::AbstractMatrix{D},
        barrier::AbstractMatrix{Bool}) where {N, T, D<:Cell{N, T}}
    ds = directions(lb)
    @inbounds for ci in CartesianIndices(newgrid)
        i, j = ci.I
        newgrid[ci] = Cell(ntuple(N) do k
            ei = ds[k]
            m, n = size(grid)
            i2, j2 = mod1(i - ei[1], m), mod1(j - ei[2], n)
            if barrier[i2, j2]
                density(grid[i, j], flip_direction_index(lb, k))
            else
                density(grid[i2, j2], k)
            end
        end)
    end
end

# collision step, applied on a single cell
function collide(lb::AbstractLBConfig{D, N}, rho::Cell; viscosity = 0.02) where {D, N}
    omega = 1 / (3 * viscosity + 0.5)   # "relaxation" parameter
    # Recompute macroscopic quantities:
    v = momentum(lb, rho)
    return (1 - omega) * rho + omega * equilibrium_density(lb, density(rho), v)
end

"""
    curl(u::AbstractMatrix{Point2D{T}})

Compute the curl of the momentum field in 2D, which is defined as:
```math
∂u_y/∂x−∂u_x/∂y
```
"""
function curl(u::Matrix{Point2D{T}}) where T 
    return map(CartesianIndices(u)) do ci
        i, j = ci.I
        m, n = size(u)
        uy = u[mod1(i + 1, m), j][2] - u[mod1(i - 1, m), j][2]
        ux = u[i, mod1(j + 1, n)][1] - u[i, mod1(j - 1, n)][1]
        return uy - ux # a factor of 1/2 is missing here?
    end
end

"""
    LatticeBoltzmann{D, N, T, CFG, MT, BT}

A lattice Boltzmann simulation with D dimensions, N velocities, and lattice configuration CFG.

### Fields
- `config::CFG`: lattice configuration
- `grid::MT`: density of the fluid
- `gridcache::MT`: cache for the density of the fluid
- `barrier::BT`: barrier configuration
"""
struct LatticeBoltzmann{D, N, T, CFG<:AbstractLBConfig{D, N}, MT<:AbstractMatrix{Cell{N, T}}, BT<:AbstractMatrix{Bool}}
    config::CFG
    grid::MT
    gridcache::MT
    barrier::BT
end
function LatticeBoltzmann(config::AbstractLBConfig{D, N}, grid::AbstractMatrix{<:Cell}, barrier::AbstractMatrix{Bool}) where {D, N}
    @assert size(grid) == size(barrier)
    return LatticeBoltzmann(config, grid, similar(grid), barrier)
end

"""
    step!(lb::LatticeBoltzmann)

Perform a single step of the lattice Boltzmann simulation.
"""
function step!(lb::LatticeBoltzmann)
    copyto!(lb.gridcache, lb.grid)
    stream!(lb.config, lb.grid, lb.gridcache, lb.barrier)
    lb.grid .= collide.(Ref(lb.config), lb.grid)
    return lb
end

"""
    example_d2q9(; height = 80, width = 200, u0 = Point(0.0, 0.1))

A D2Q9 lattice Boltzmann simulation example. A simple linear barrier is added to the lattice.

### Arguments
- `height::Int`: height of the lattice
- `width::Int`: width of the lattice
- `u0::Point2D`: initial and in-flow speed
"""
function example_d2q9(; 
        height = 80,                       # lattice dimensions
        width = 200,
        u0 = Point(0.0, 0.1)                           # initial and in-flow speed
    )
    # Initialize all the arrays to steady rightward flow:
    rho = equilibrium_density(D2Q9(), 1.0, u0)
    rgrid = fill(rho, height, width)

    # Initialize barriers:
    barrier = falses(height, width)                          # True wherever there's a barrier
    mid = div(height, 2)
    barrier[mid-8:mid+8, div(height,2)] .= true              # simple linear barrier

    return LatticeBoltzmann(D2Q9(), rgrid, barrier)
end