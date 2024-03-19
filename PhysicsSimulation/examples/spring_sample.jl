M = C = 1.0
C_matrix = [-C C 0 0 0; C -2C C 0 0; 0 C -2C C 0; 0 0 C -2C C; 0 0 0 C -C]
@info "We have the following C matrix: $C_matrix, and the mass matrix is $M, $C is the stiffness."

using LinearAlgebra
evals, evecs = eigen(C_matrix)
@info "The eigenvalues are $evals, and the eigenvectors are $evecs."

second_omega = sqrt(-evals[2]/M)
second_mode = evecs[:, 2]
@info "The second mode is $second_mode, and the second omega is $second_omega."

L = size(C_matrix, 1)
phi0 = zeros(L)
u(t) = (0:L-1) .+ second_mode .* cos.(-second_omega .* t)
@info "Length of C_matrix is $L, and the initial condition is $phi0."

u(1.0)
L = 5
u0 = second_mode .* cos.(phi0)
@info "The initial condition is $u0, and the length of the chain is $L."

using PhysicsSimulation
dt, nsteps = 0.1, round(10 * 2π/second_omega)
spring = spring_chain(u0, C, M; periodic=false)
states = leapfrog_simulation(spring; dt, nsteps)
@info "dt is $dt, and the number of steps is $nsteps."

using Makie, CairoMakie
p2(x::PhysicsSimulation.Point{1}) = Point2f(x.data[1], 0.0)
getcoos(b::LeapFrogSystem) = p2.(coordinate(b.sys))
locs = getcoos.(states)
locs2 = [Point2f.(u(t), -0.1) for t in dt * (0:nsteps)]
@info "The coordinates are $getcoos, The coordinates are $locs, and the exact coordinates are $locs2."


# visualize the system
fig = Figure()
ax1 = Axis(fig[1, 1]; limits=(-1, length(locs[1]), -0.4, 0.3))
coos1 = Observable(locs[1])
coos2 = Observable(locs2[1])
scatter!(ax1, coos1, markersize = 10, color = :blue, limits = (-1, length(locs[1]), -1, 1), label="simulation")
scatter!(ax1, coos2, markersize = 10, color = :cyan, limits = (-1, length(locs[1]), -1, 1), label="exact")
axislegend(ax1)
@info "The first coordinates are $coos1, and the second coordinates are $coos2, and plot axislegend to visualize the system."

record(fig, joinpath(@__DIR__, "springs-demo.gif"), 2:length(states); framerate = 24) do i
    coos1[] = locs[i]
    coos2[] = locs2[i]
end
@info "The gif is named springs-demo.gif, and it is recorded in examples/spring-demo.gif."

