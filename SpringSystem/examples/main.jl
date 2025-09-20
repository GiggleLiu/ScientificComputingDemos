using CairoMakie
using CairoMakie: RGBA
using SpringSystem
using SpringSystem: eigenmodes, eigensystem, nv
using LinearAlgebra

function run_spring_chain(; C = 3.0, M = 1.0, L = 20, u0 = 0.2 * randn(L))
    # setup the spring chain model
    spring = spring_chain(u0, C, M; periodic=false)

    @info """Setup spring chain model:
    - mass = $M
    - stiffness = $C
    - length of chain = $L
    """

    # simulate the spring chain with leapfrog sympletic integrator
    @info """Simulating with leapfrog sympletic integrator:
    - dt = 0.1
    - number of steps = 500
    """
    simulated = leapfrog_simulation(spring; dt=0.1, nsteps=500)

    # solve the spring system exactly with eigenmodes
    @info """Soving the spring system exactly with eigenmodes"""
    exact = waveat(eigenmodes(eigensystem(spring)), u0, 0.1 * (0:500))

    return simulated, exact
end

function visualize_simulation(simulated, exact)
    # helper functions to visualize the system
    p2(x::SpringSystem.Point{1}) = Point2f(x.data[1], 0.0)
    getcoos(b::LeapFrogSystem) = p2.(coordinate(b.sys))
    getendpoints(b::LeapFrogSystem) = p2.(b.a)

    L = nv(simulated[1].sys.topology)
    locs = getcoos.(simulated)
    locs2 = [map(x->Point2f(x, -0.05), (0:L-1) .+ x) for x in exact]
    vecs = getendpoints.(simulated)

    fig = Figure()
    ax1 = Axis(fig[1, 1]; limits=(-1, length(locs[1]), -0.4, 0.3))
    coos1 = Observable(locs[1])
    coos2 = Observable(locs2[1])
    endpoints = Observable(vecs[1])
    scatter!(ax1, coos1, markersize = 10, color = :blue, label="simulation")
    arrows2d!(ax1, coos1, endpoints; color = :red)
    scatter!(ax1, coos2, markersize = 10, color = :cyan, label="exact")
    axislegend(ax1)

    filename = joinpath(@__DIR__, "springs-simulate.mp4")
    record(fig, filename, 2:length(simulated); framerate = 24) do i
        coos1[] = locs[i]
        coos2[] = locs2[i]
        endpoints[] = vecs[i]
    end
    @info "Animation stored to: `$filename`"
end

# visualize different eigenmodes
function visualize_eigenmodes(c::SpringModel)
    sys = eigensystem(c)
    modes = eigenmodes(sys)
    L = nv(c.topology)

    # compute the locations of the wave modes using the eigenmodes
    locations(idx::Int, t) = Point2f.((0:L-1) .+ waveat(modes, idx, t), 0.0)

    fig = Figure()
    coos = Observable[]
    indices = [1, 5, 10, 15]
    for (k, idx) in enumerate(indices)
        ax = Axis(fig[k, 1])
        push!(coos, Observable(locations(idx, 0.0)))
        scatter!(ax, coos[end], markersize = 10, color = :blue, label="ω = $(round(modes.frequency[idx], digits=3))")
        axislegend(ax)
    end
    @info "Create a figure with multiple scatter plots, each representing the locations of wave modes at different indices over time."

    filename = joinpath(@__DIR__, "springs.mp4")
    record(fig, filename, 2:500; framerate = 24) do i
        t = 0.1 * i
        for (k, idx) in enumerate(indices)
            coos[k][] = locations(idx, t)
        end
    end
    @info "Make a video, and it is recorded in `$filename`."
end

# run the simulation
simulated, exact = run_spring_chain()
visualize_simulation(simulated, exact)
visualize_eigenmodes(simulated[1].sys)