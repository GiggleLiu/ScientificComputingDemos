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