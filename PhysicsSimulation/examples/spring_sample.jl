M = C = 1.0
C_matrix = [-C C 0 0 0; C -2C C 0 0; 0 C -2C C 0; 0 0 C -2C C; 0 0 0 C -C]
@info "Define the mass matrix is $M, and $C is the stiffness."

using LinearAlgebra
evals, evecs = eigen(C_matrix)
@info "Define the eigenvalues and the eigenvectors."

second_omega = sqrt(-evals[2]/M)
second_mode = evecs[:, 2]
@info "Set some parameters."

L = size(C_matrix, 1)
phi0 = zeros(L)
u(t) = (0:L-1) .+ second_mode .* cos.(-second_omega .* t)
@info "Length of C_matrix is $L, and set the zero initial condition."

u(1.0)
L = 5
u0 = second_mode .* cos.(phi0)
@info "Define the length of the chain, and the initial condition."

using PhysicsSimulation
dt, nsteps = 0.1, round(10 * 2π/second_omega)
spring = spring_chain(u0, C, M; periodic=false)
states = leapfrog_simulation(spring; dt, nsteps)

using Makie, CairoMakie
p2(x::PhysicsSimulation.Point{1}) = Point2f(x.data[1], 0.0)
getcoos(b::LeapFrogSystem) = p2.(coordinate(b.sys))
locs = getcoos.(states)
locs2 = [Point2f.(u(t), -0.1) for t in dt * (0:nsteps)]
@info "using Makie, CairoMakie, and set steps and other parameters."


# visualize the system
fig = Figure()
ax1 = Axis(fig[1, 1]; limits=(-1, length(locs[1]), -0.4, 0.3))
coos1 = Observable(locs[1])
coos2 = Observable(locs2[1])
scatter!(ax1, coos1, markersize = 10, color = :blue, limits = (-1, length(locs[1]), -1, 1), label="simulation")
scatter!(ax1, coos2, markersize = 10, color = :cyan, limits = (-1, length(locs[1]), -1, 1), label="exact")
axislegend(ax1)
@info "Plot figure and set the scatter, axis to visualize the system."

record(fig, joinpath(@__DIR__, "springs-demo.gif"), 2:length(states); framerate = 24) do i
    coos1[] = locs[i]
    coos2[] = locs2[i]
end
@info "The gif is finished, and it is recorded in examples/spring-demo.gif."



