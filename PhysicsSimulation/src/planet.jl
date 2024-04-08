module Bodies
import LinearAlgebra
using ..PhysicsSimulation: Point3D, Point
import ..PhysicsSimulation
using CSV, DataFrames

# NOTE:
# unit of time -> year
# unit of space -> AU
const year = 3.154e7 #year in seconds
const AU = 1.496e11 #in m

const mass_solar = 1.988544e30 # in kg
const G_standard = 6.67259e-11 # in m^3/(kg-s^2)
const G_year_AU = G_standard * (1 / AU)^3 / (1 / mass_solar * (1 / year)^2)
const dayToYear = 365.25

struct Body{D, T}
    r::Point{D, T}
    v::Point{D, T}
    m::T
end

solar_system_data() = CSV.read(joinpath(pkgdir(PhysicsSimulation), "data", "solar_system.csv"), DataFrame)
solar_system() = newton_system_from_data(solar_system_data())

# mass: kg
function newton_system_from_data(data)
    map(eachrow(data)) do planet
        r = Point(planet.x, planet.y, planet.z)
        v = Point(planet.vx, planet.vy, planet.vz) * dayToYear
        m = planet.mass / mass_solar
        Body(r, v, m)
    end |> NewtonSystem
end

abstract type AbstractHamiltonianSystem{D} end
struct NewtonSystem{D, T} <: AbstractHamiltonianSystem{D}
    bodies::Vector{Body{D, T}}
end
coordinate(b::NewtonSystem) = [b.bodies[i].r for i in 1:length(b.bodies)]
coordinate(b::NewtonSystem, i::Int) = b.bodies[i].r
function offset_coordinate!(b::NewtonSystem, i::Int, val)
    b.bodies[i] = Body(b.bodies[i].r + val, b.bodies[i].v, b.bodies[i].m)
end
function offset_velocity!(b::NewtonSystem, i::Int, val)
    b.bodies[i] = Body(b.bodies[i].r, b.bodies[i].v + val, b.bodies[i].m)
end
velocity(b::NewtonSystem) = [b.bodies[i].v for i in 1:length(b.bodies)]
velocity(b::NewtonSystem, i::Int) = b.bodies[i].v
mass(b::NewtonSystem) = [b.bodies[i].m for i in 1:length(b.bodies)]
mass(b::NewtonSystem, i::Int) = b.bodies[i].m
Base.length(bds::NewtonSystem) = length(bds.bodies)

end


using .Bodies: G_year_AU, Body, solar_system, NewtonSystem, AbstractHamiltonianSystem
import .Bodies: coordinate, velocity, offset_coordinate!, offset_velocity!, mass

function energy(bds::NewtonSystem{T}) where T
    eng = zero(T)
    # kinetic energy
    for p in bds.bodies
        eng += p.m * norm2(p.v) / 2
    end
    # potential energy
    for j in 1:length(bds)
        pj = bds.planets[j]
        for k in j+1:bds.nplanets
            pk = bds.planets[k]
            eng -= G_year_AU * pj.m * pk.m / sqdist(pj.r, pk.r)
        end
    end
    eng
end

function barycenter(m, mTot, coo::Point3D) # Find Barycenter
    #    m : mass of planet
    # mTot : total mass of system
    #  coo : coordinates of planet
    #
    return (m * coo) / mTot
end

function momentum(body::Body)
    return m * cross(body.r, body.v)
end

@inline function acceleration(ra, rb, mb)
    d = distance(ra, rb)
    (G_year_AU * mb / d^3) * (rb - ra)
end

function update_acceleration!(a::AbstractVector{Point{D, T}}, bds::NewtonSystem) where {D, T}
    @assert length(a) == length(bds)
    @inbounds for j = 1:length(bds)
        a[j] = zero(Point{D, T})
        for k = 1:length(bds)
            j != k && (a[j] += acceleration(coordinate(bds, j), coordinate(bds, k), mass(bds, k)))
        end
    end
    return a
end

struct LeapFrogSystem{T, D, SYS<:AbstractHamiltonianSystem{D}}
    sys::SYS
    a::Vector{Point{D, T}}
    function LeapFrogSystem(bds::AbstractHamiltonianSystem, a::Vector{Point{D, T}}) where {T, D}
        @assert length(bds) == length(a)
        new{T, D, typeof(bds)}(bds, a)
    end
end
function LeapFrogSystem(bds::AbstractHamiltonianSystem)
    LeapFrogSystem(bds, zero(coordinate(bds)))
end

function step!(bdsc::LeapFrogSystem{T}, dt) where T
    sys, a = bdsc.sys, bdsc.a
    @inbounds for j = 1:length(sys)
        drj = dt / 2 * velocity(sys, j)
        offset_coordinate!(sys, j, drj)
    end
    update_acceleration!(a, sys)
    @inbounds for j = 1:length(sys)
        dvj = dt * a[j]
        offset_velocity!(sys, j, dvj)
        drj = dt / 2 * velocity(sys, j)
        offset_coordinate!(sys, j, drj)
    end
    return bdsc
end

function leapfrog_simulation(sys::AbstractHamiltonianSystem; dt, nsteps)
    cached_system = LeapFrogSystem(deepcopy(sys))
    states = [deepcopy(cached_system)]
    for i=1:nsteps
        cached_system = step!(cached_system, dt)
        push!(states, deepcopy(cached_system))
    end
    return states
end