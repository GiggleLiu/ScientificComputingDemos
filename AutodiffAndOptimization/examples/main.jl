using CairoMakie
using Printf
using AutodiffAndOptimization

bestx, bestf, history = simplex(rosenbrock, [-1.2, -1.0]; tol=1e-3)
bestx, bestf

x = -2:0.02:2
y = -2:0.02:2
f = [rosenbrock((a, b)) for b in y, a in x]

bestx, bestf, history = simplex(rosenbrock, [-1.2, -1.0]; tol=1e-3)
@info "converged in $(length(history)) steps, with error $bestf"

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="x₁", ylabel="x₂", limits=(-2, 2, -2, 2))
heatmap!(ax, x, y, log.(f))
triangles = [[[item[:,1]..., item[1,1]], [item[:,2]..., item[1, 2]]] for item in history]
triangle_x = Observable(triangles[1][1])
triangle_y = Observable(triangles[1][2])
lines!(ax, triangle_x, triangle_y; label="", color="white")
txt = Observable("step = 0")
text!(ax, -1.5, 1.5; text=txt, color=:white, fontsize=20)

filename = joinpath(@__DIR__, "simplex.mp4")
record(fig, filename, 1:length(triangles); framerate = 24) do i
    txt[] = @sprintf "step = %d, f = %.2e" i minimum(k->rosenbrock(history[i][k,:]), 1:3)
    "step = $(i-1), f = $(rosenbrock(history[i]))"
    triangle_x[] = triangles[i][1]
    triangle_y[] = triangles[i][2]
end
