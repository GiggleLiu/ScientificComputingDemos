using SpinDynamics, Graphs, Profile

function simulate_afm_grid(n::Int; nsteps::Int=100, dt::Float64=0.1, checkpoint_steps::Int=10)
    topology = grid((n, n))
    sys = ClassicalSpinSystem(topology, -ones(ne(topology)))
    spins = random_spins(nv(topology))
    _, history = @profile simulate!(spins, sys; nsteps=nsteps, dt=dt, checkpoint_steps=checkpoint_steps, algorithm=TrotterSuzuki{2}(topology))
    return history
end

history = simulate_afm_grid(50, nsteps=10000, dt=0.01)
Profile.print(format=:flat, mincount=100)