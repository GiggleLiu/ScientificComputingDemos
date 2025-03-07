"""
    LeapFrogSystem{T, D, SYS<:AbstractHamiltonianSystem{D}}

The leapfrog system is a symplectic integrator for the Hamiltonian system.

### Fields
- `sys` is the Hamiltonian system
- `a` is the acceleration of the system
"""
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

# evolve the Hamiltonian system with the leapfrog method for a time step dt
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

# evolve the Hamiltonian system with the leapfrog method for nsteps * dt time, and return the states
function leapfrog_simulation(sys::AbstractHamiltonianSystem; dt, nsteps)
    cached_system = LeapFrogSystem(deepcopy(sys))
    states = [deepcopy(cached_system)]
    for i=1:nsteps
        cached_system = step!(cached_system, dt)
        push!(states, deepcopy(cached_system))
    end
    return states
end