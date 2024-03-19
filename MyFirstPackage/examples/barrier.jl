using Makie: RGBA
using Makie, CairoMakie
using MyFirstPackage

# simulate a fluid with a barrier
lb = example_d2q9()
states = [copy(lb.grid)]
for i=1:2000
    step!(lb)
    i % 20 == 0 && push!(states, copy(lb.grid))
end
curls = [curl(momentum.(Ref(lb.config), s)) for s in states]

# Set up the visualization with Makie:
vorticity = Observable(curls[1]')
fig, ax, plot = image(vorticity, colormap = :jet, colorrange = (-0.1, 0.1))

# Add barrier visualization:
barrier_img = map(x -> x ? RGBA(0, 0, 0, 1) : RGBA(0, 0, 0, 0), lb.barrier)
image!(ax, barrier_img')

using BenchmarkTools
@benchmark step!($(deepcopy(lb)))

record(fig, joinpath(@__DIR__, "barrier.mp4"), 1:100; framerate = 10) do i
    vorticity[] = curls[i+1]'
end