using Makie: RGBA
using Makie, CairoMakie
using PhysicsSimulation
using Enzyme

# helper function to visualize the system
getcoo(b::Body) = Point3f(b.r.data)
getcoos(b::LeapFrogSystem) = getcoo.(b.sys.bodies)
getarrows(b) = [Point3f(x.data) for x in b.a]
function make_movie(filename::String, states, color)
    fig = Figure()
    ax = Axis3(fig[1, 1]; limits=(-30, 30, -30, 30, -30, 30))
    coos = Observable(getcoos(states[1]))    # position
    endpoints = Observable(getarrows(states[1])) # acceleration
    scatter!(ax, coos; markersize = 10, color)
    record(fig, filename, 2:10:length(states); framerate = 24) do i
        coos[] = getcoos(states[i])
        endpoints[] = getarrows(states[i])
    end
    @info "Recording saved to file: $filename"
end

data = PhysicsSimulation.Bodies.solar_system_data()
@info "Loading solar system data: $data"
const solar = PhysicsSimulation.Bodies.newton_system_from_data(data)
const Δt, num_steps = 0.01, 1000
@info "simulate the solar system with time step $Δt and $num_steps steps"
states = leapfrog_simulation(solar; dt=Δt, nsteps=num_steps)

make_movie(joinpath(@__DIR__, "solar-system.mp4"), states, [k == 4 ? :yellow : :blue for k in 1:length(solar)])

# modify the solar system by adding a stone on Pluto
function modified_solar_system(v0)
    # I throw a stone on Pluto, with velocity v0
    newbody = Body(solar.bodies[end].r + PhysicsSimulation.Point(0.01, 0.0, 0.0), v0, 1e-16)
    bds = copy(solar.bodies)
    push!(bds, newbody)
    return NewtonSystem(bds)
end

function loss_hit_earth(v0)
    # simulate the system
    lf = LeapFrogSystem(modified_solar_system(PhysicsSimulation.Point(v0...)))
    for _ = 1:num_steps
        step!(lf, Δt)
    end
    return PhysicsSimulation.distance(lf.sys.bodies[end].r, lf.sys.bodies[4].r)  # final distance to earch
end
v0 = solar.bodies[end].v * 2
y0 = loss_hit_earth(v0)

@info "Now I throw a stone on Pluto with velocity $v0, and the final distance to earth is $y0."

msolar0 = modified_solar_system(v0)
states0 = leapfrog_simulation(msolar0; dt=Δt, nsteps=num_steps)
color = [k == length(msolar0) ? :red : (k == 4 ? :yellow : :blue) for k in 1:length(msolar0)]
make_movie(joinpath(@__DIR__, "solar-system-hit-earth-beforeopt.mp4"), states0, color)

# gradient of the loss function, obtained by Enzyme
function gradient_hit_earth!(g, v)
    g .= Enzyme.autodiff(Enzyme.Reverse, loss_hit_earth, Active, Active(PhysicsSimulation.Point(v...)))[1][1]
    g
end

# verify the gradient
using Test, FiniteDifferences

@testset "grad" begin
    enzyme_gradient = gradient_hit_earth!(zeros(3), v0)
    finitediff_gradient, = FiniteDifferences.jacobian(central_fdm(5,1), loss_hit_earth, v0)
    @info "The gradient of the loss function is $enzyme_gradient, and the finite difference gradient is $finitediff_gradient."
    @test sum(abs2, enzyme_gradient - finitediff_gradient') < 1e-6
end

# what is the velocity that hits the earth?
using Optim
function hit_earth(v0)
    # optimize the velocity, such that the stone hits the earth
    # the optimizer is LBFGS
    return Optim.optimize(loss_hit_earth, gradient_hit_earth!, [v0...], LBFGS()).minimizer
end

vopt = hit_earth(v0)
yopt = loss_hit_earth(vopt)
@info "The optimized velocity is $vopt, and the final distance to earth is $yopt."

msolar = modified_solar_system(PhysicsSimulation.Point(vopt...))
states = leapfrog_simulation(msolar; dt=Δt, nsteps=num_steps)
make_movie(joinpath(@__DIR__, "solar-system-hit-earth.mp4"), states, color)

# orbitals
fig = Figure()
ax = Axis3(fig[1, 1]; limits=(-30, 30, -30, 30, -30, 30))
for k=1:length(msolar)
    orbit = [getcoo(states[i].sys.bodies[k]) for i in 1:length(states)]
    scatter!(ax, orbit; markersize = 10, color=color[k])
end
fig
filename = joinpath(@__DIR__, "planet_orbitals.png")
save(filename, fig)
@info "Plot the orbitals of the planets in the solar system, saved to $filename"
