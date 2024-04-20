using CairoMakie: RGBA
using CairoMakie
using LatticeBoltzmannModel

using CUDA

# simulate a fluid with a barrier
lb = CUDA.cu(example_d2q9())
states = [copy(lb.grid)]
for i=1:2000
    step!(lb)
    i % 20 == 0 && push!(states, copy(lb.grid))
end
curls = [curl(Matrix(momentum.(Ref(lb.config), s))) for s in states]

# Set up the visualization with Makie:
vorticity = Observable(curls[1]')
fig, ax, plot = image(vorticity, colormap = :jet, colorrange = (-0.1, 0.1))

# Add barrier visualization:
barrier_img = map(x -> x ? RGBA(0, 0, 0, 1) : RGBA(0, 0, 0, 0), lb.barrier)
image!(ax, barrier_img')

using BenchmarkTools
@benchmark step!($(deepcopy(lb)))

CairoMakie.record(fig, joinpath(@__DIR__, "barrier_gpu.mp4"), 1:100; framerate = 10) do i
    vorticity[] = curls[i+1]'
end
