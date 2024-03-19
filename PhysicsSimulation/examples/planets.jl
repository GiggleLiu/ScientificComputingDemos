using Makie: RGBA
using Makie, CairoMakie
using PhysicsSimulation

states = leapfrog_simulation(solar_system(); dt=0.01, nsteps=5500)

# visualize the system
getcoo(b::Body) = Point3f(b.r.data)
getcoos(b::LeapFrogSystem) = getcoo.(b.sys.bodies)
coos = Observable(getcoos(states[1]))
getarrows(b) = [Point3f(x.data) for x in b.a]
endpoints = Observable(getarrows(states[1]))
fig, ax, plot = scatter(coos, markersize = 10, color = :blue, limits = (-50, 50, -50, 50, -50, 50))
arrows!(ax, coos, endpoints; color = :red)

# movie
fig = Figure()
ax = Axis3(fig[1, 1]; limits=(-30, 30, -30, 30, -30, 30))
scatter!(ax, coos, markersize = 10, color = :blue)
record(fig, joinpath(@__DIR__, "planet.mp4"), 2:10:length(states); framerate = 24) do i
    coos[] = getcoos(states[i])
    endpoints[] = getarrows(states[i])
end

# orbitals
fig = Figure()
ax = Axis3(fig[1, 1]; limits=(-30, 30, -30, 30, -30, 30))
for k=1:length(solar_system())
    orbit = [getcoo(states[i].sys.bodies[k]) for i in 1:length(states)]
    lines!(ax, orbit, markersize = 10, color = :blue)
end
fig
save(joinpath(@__DIR__, "planet_orbitals.png"), fig)