using Test
using Spinglass: ClassicalSpinSystem, SpinVector, simulate!, greedy_coloring, is_valid_coloring, partite_edges
using Graphs

@testset "Spin dynamics" begin
    sys = ClassicalSpinSystem(SimpleGraph(3), [1.0, 1.0, 1.0])
    spins = [SpinVector(ntuple(i -> randn(), 3)) for _ in 1:nv(sys.topology)]
    simulate!(spins, sys; nsteps=10, dt=0.1)
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