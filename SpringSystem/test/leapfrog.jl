using Test
using SpringSystem
using LinearAlgebra

# Simple harmonic oscillator as a test system
struct SimpleHarmonianOscillator{D} <: SpringSystem.AbstractHamiltonianSystem{D}
    position::Vector{Point{D, Float64}}
    velocity::Vector{Point{D, Float64}}
    k::Float64  # spring constant
    m::Float64  # mass
end

function SimpleHarmonianOscillator(; k=1.0, m=1.0)
    # Single particle in 1D
    pos = [Point(1.0)]  # Initial position x=1
    vel = [Point(0.0)]  # Initial velocity v=0
    SimpleHarmonianOscillator{1}(pos, vel, k, m)
end

Base.length(sys::SimpleHarmonianOscillator) = length(sys.position)
SpringSystem.coordinate(sys::SimpleHarmonianOscillator) = sys.position
SpringSystem.velocity(sys::SimpleHarmonianOscillator, i::Int) = sys.velocity[i]
SpringSystem.offset_coordinate!(sys::SimpleHarmonianOscillator, i::Int, dr) = sys.position[i] += dr
SpringSystem.offset_velocity!(sys::SimpleHarmonianOscillator, i::Int, dv) = sys.velocity[i] += dv

function SpringSystem.update_acceleration!(a::Vector{<:Point}, sys::SimpleHarmonianOscillator)
    # F = -kx for simple harmonic oscillator
    for i in 1:length(sys)
        a[i] = Point(-sys.k/sys.m * sys.position[i][1])
    end
end

@testset "LeapFrog Integration" begin
    # Test system construction
    sys = SimpleHarmonianOscillator()
    leapfrog_sys = LeapFrogSystem(sys)
    @test length(leapfrog_sys.a) == length(sys)
    @test leapfrog_sys.sys === sys

    # Test single step
    dt = 0.1
    stepped_sys = SpringSystem.step!(leapfrog_sys, dt)
    @test stepped_sys === leapfrog_sys  # Should modify in-place
    @test stepped_sys.sys.position[1][1] != 1.0  # Position should change

    # Test simulation
    dt = 0.1
    nsteps = 100
    states = leapfrog_simulation(sys; dt=dt, nsteps=nsteps)
    @test length(states) == nsteps + 1
    
    # Test energy conservation (approximately)
    function total_energy(state::LeapFrogSystem)
        sys = state.sys
        # E = 1/2 mv² + 1/2 kx²
        kinetic = 0.5 * sys.m * sum(v[1]^2 for v in sys.velocity)
        potential = 0.5 * sys.k * sum(x[1]^2 for x in sys.position)
        return kinetic + potential
    end
    
    initial_energy = total_energy(states[1])
    for state in states[2:end]
        current_energy = total_energy(state)
        # Check energy conservation with some tolerance
        @test isapprox(current_energy, initial_energy, rtol=1e-2)
    end
    
    # Test periodicity (approximately)
    # For simple harmonic oscillator, period T = 2π√(m/k)
    period = 2π * sqrt(sys.m/sys.k)
    n_period_steps = round(Int, period/dt)
    
    if n_period_steps <= nsteps
        initial_pos = states[1].sys.position[1][1]
        period_pos = states[n_period_steps+1].sys.position[1][1]
        @test isapprox(initial_pos, period_pos, rtol=1e-1)
    end
end
