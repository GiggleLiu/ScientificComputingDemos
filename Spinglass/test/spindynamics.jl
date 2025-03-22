using Test
using Spinglass: ClassicalSpinSystem, SpinVector, simulate!, greedy_coloring, is_valid_coloring, partite_edges, single_spin_dynamics_operator, single_spin_dynamics, SVector, TrotterSuzuki
using Graphs

@testset "Spin dynamics" begin
    topology = grid((3, 3))
    sys = ClassicalSpinSystem(topology, [1.0, 1.0, 1.0])
    spins = [SVector(ntuple(i -> randn(), 3)) for _ in 1:nv(sys.topology)]
    state, history = simulate!(spins, sys; nsteps=100, dt=0.1, checkpoint_steps=10, algorithm=TrotterSuzuki{2}(topology))
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
