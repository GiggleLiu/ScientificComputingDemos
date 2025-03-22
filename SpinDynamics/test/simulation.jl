using Test
using SpinDynamics: ClassicalSpinSystem, simulate!, greedy_coloring, is_valid_coloring, partite_edges, single_spin_dynamics_operator, single_spin_dynamics, SVector, TrotterSuzuki, random_spins, TimeDependent, SVector
using Graphs

@testset "Spin dynamics" begin
    topology = grid((3, 3))
    sys = ClassicalSpinSystem(topology, fill(SVector(1.0, 1.0, 1.0), ne(topology)))
    spins = [SVector(1.0, 0.0, 0.0) for _ in 1:nv(sys.topology)]
    state, history = simulate!(spins, sys; nsteps=100, dt=0.1, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    @test state ≈ [SVector(1.0, 0.0, 0.0) for _ in 1:nv(sys.topology)]
    @test length(history) == 10
    @test length(state) == 9
end

@testset "Greedy coloring" begin
    g = grid((10, 10))
    coloring = greedy_coloring(g)
    @test is_valid_coloring(g, coloring)
    @test length(unique(coloring)) <= 5
    eparts = partite_edges(g)
    @test length(eparts) <= 4

    g = path_graph(10)
    eparts = partite_edges(g)
    @test length(eparts) <= 2
end

@testset "single spin dynamics" begin
    #s = SVector(randn(), randn(), randn())
    s = SVector(1.0, 0.0, 0.0)
    field = SVector(randn(), randn(), randn())
    op = single_spin_dynamics_operator(field)
    @test op * s ≈ single_spin_dynamics(field, s)
end

@testset "time dependent" begin
    J = TimeDependent(fill(SVector(0.0, 0.0, 0.0), ne(grid((3, 3)))), (J, t) -> (J .= Ref(SVector(0.0, 0.0, 0.0))))
    h = TimeDependent(fill(SVector(0.0, 0.0, 0.0), nv(grid((3, 3)))), (h, t) -> (h .= Ref(t * SVector(0.0, 0.0, 1.0))))
    topology = grid((3, 3))
    sys = ClassicalSpinSystem(topology, J; bias=h)
    spins = [SVector(0.0, 0.0, 1.0) for _ in 1:nv(sys.topology)]
    E0 = energy(SpinDynamics.instantiate(sys, 0.0), spins)
    @test E0 ≈ 0.0 atol=1e-6
    state, history = simulate!(spins, sys; nsteps=100, dt=0.1, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    E1 = energy(SpinDynamics.instantiate(sys, 10.0), state)
    @test E1 ≈ 90.0 atol=1e-6


    h = TimeDependent(fill(SVector(0.0, 0.0, 0.0), nv(grid((3, 3)))), (h, t) -> (h .= Ref(SVector(t * -1.0/10, 0.0, -1.0/10 * (10 - t)))))
    sys = ClassicalSpinSystem(topology, J; bias=h)
    spins = [SVector(0.0, 0.0, 1.0) for _ in 1:nv(sys.topology)]
    E0 = energy(SpinDynamics.instantiate(sys, 0.0), spins)
    E0b = energy(SpinDynamics.instantiate(sys, 10.0), spins)
    @test E0 ≈ -9.0 atol=1e-6
    @test E0b ≈ 0.0 atol=1e-6
    state, history = simulate!(spins, sys; nsteps=100, dt=0.1, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    @test energy(SpinDynamics.instantiate(sys, 10.0), state) ≈ -9.0 atol=1e-6
    @test energy(SpinDynamics.instantiate(sys, 0.0), state) ≈ 0.0 atol=1e-2
end

@testset "damping" begin
    # dampling drives the system to the ground state
    topology = grid((3, 3))
    J = fill(SVector(1.0, 1.0, 1.0), ne(topology))
    h = fill(SVector(0.0, 0.0, 0.0), nv(topology))
    sys_damped = ClassicalSpinSystem(topology, J; bias=h, damping=2.0)
    spins = [SVector(randn(), randn(), randn()) |> normalize for _ in 1:nv(sys_damped.topology)]
    state, history = simulate!(spins, sys_damped; nsteps=100, dt=0.1, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
    @test energy(sys_damped, state) ≈ -12.0 atol=1e-3
end