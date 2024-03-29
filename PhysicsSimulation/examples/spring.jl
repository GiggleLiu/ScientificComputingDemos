using Makie: RGBA
using Makie, CairoMakie
using PhysicsSimulation


@info """Setup spring chain model:
- mass = $M
- stiffness = $C
- length of chain = $L
"""
C, M = 3.0, 1.0
L = 20
u0 = 0.2 * randn(L)
spring = spring_chain(u0, C, M; periodic=false)

@info """Simulating with leapfrog sympletic integrator:
- dt = 0.1
- number of steps = 500
"""
states = leapfrog_simulation(spring; dt=0.1, nsteps=500)

@info """Soving the spring system exactly with eigenmodes"""
states_exact = waveat(eigenmodes(eigensystem(spring)), u0, 0.1 * (0:500))

# visualize the system
p2(x::PhysicsSimulation.Point{1}) = Point2f(x.data[1], 0.0)
getcoos(b::LeapFrogSystem) = p2.(coordinate(b.sys))
getendpoints(b::LeapFrogSystem) = p2.(b.a)

locs = getcoos.(states)
locs2 = [map(x->Point2f(x, -0.05), (0:L-1) .+ x) for x in states_exact]
vecs = getendpoints.(states)

fig = Figure()
ax1 = Axis(fig[1, 1]; limits=(-1, length(locs[1]), -0.4, 0.3))
coos1 = Observable(locs[1])
coos2 = Observable(locs2[1])
endpoints = Observable(vecs[1])
scatter!(ax1, coos1, markersize = 10, color = :blue, limits = (-1, length(locs[1]), -1, 1), label="simulation")
arrows!(ax1, coos1, endpoints; color = :red)
scatter!(ax1, coos2, markersize = 10, color = :cyan, limits = (-1, length(locs[1]), -1, 1), label="exact")
axislegend(ax1)

filename = joinpath(@__DIR__, "springs-simulate.mp4")
record(fig, filename, 2:length(states); framerate = 24) do i
    coos1[] = locs[i]
    coos2[] = locs2[i]
    endpoints[] = vecs[i]
end
@info "Animation stored to: `$filename`"

@info "Visualizing the eigenmodes"
# visualize eigenmodes
c = spring_chain(zeros(L), C, M; periodic=false)
sys = eigensystem(c)
modes = eigenmodes(sys)
@info "The stiffness is $C, the mass is $M, and the length of the chain is $L, and define the eigenmodes"

# wave function
locations(idx::Int, t) = Point2f.((0:L-1) .+ waveat(modes, idx, t), 0.0)

fig = Figure()
coos = Observable[]
for (k, idx) in enumerate([1, 5, 10, 15])
    ax = Axis(fig[k, 1])
    push!(coos, Observable(locations(idx, 0.0)))
    scatter!(ax, coos[end], markersize = 10, color = :blue, limits = (-1, length(locs[1]), -1, 1))
end
@info "Create a figure with multiple scatter plots, each representing the locations of wave modes at different indices over time."

record(fig, joinpath(@__DIR__, "springs.mp4"), 2:length(states); framerate = 24) do i
    t = 0.1 * i
    for (k, idx) in enumerate([2, 5, 10, 15])
        coos[k][] = locations(idx, t)
    end
end
@info "Make a gif, and it is recorded in examples/springs.mp4."