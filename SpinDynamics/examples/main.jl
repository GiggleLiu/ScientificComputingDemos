using SpinDynamics, Graphs, SpinDynamics.StaticArrays

function simulate_afm_grid(n::Int)
    topology = grid((n, n))
    sys = ClassicalSpinSystem(topology, ones(ne(topology)), fill(SVector(1.0, 0.0, 0.0), nv(topology)))
    spins = random_spins(nv(topology); xbias=5.0)
    _, history = simulate!(spins, sys; nsteps=1000, dt=0.01, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    return history
end

history = simulate_afm_grid(5)
visualize_spins_animation(vec([(i, j, 0) for i in 1:5 for j in 1:5]), history; filename=joinpath(@__DIR__, "spin_animation.mp4"))

function twopoint_bifurcation_model()
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    sys = SimulatedBifurcation(1.0, 1.0, g)
    state = SimulatedBifurcationState(randn(nv(g)), randn(nv(g)))
    simulate!(state, sys; nsteps=1000, dt=0.01, checkpoint_steps=10)
    return state, sys
end

