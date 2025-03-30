using SpinDynamics, Test, Graphs

@testset "SimulatedBifurcation" begin
    g = smallgraph(:petersen)
    for KIND in (:aSB, :bSB, :dSB)
        sys = SimulatedBifurcation{KIND}(1.0, 0.2, g, randn(ne(g)))
        x = randn(nv(g))
        f = SpinDynamics.force(sys, x)
        δx = randn(nv(g)) * 1e-5
        engdiff = SpinDynamics.potential_energy(sys, x + δx/2) - SpinDynamics.potential_energy(sys, x - δx/2)
        @test isapprox(sum(f .* δx), -engdiff, rtol=1e-3)
    end
    @test SimulatedBifurcation{:aSB}(g, randn(ne(g))) isa SimulatedBifurcation{Float64, :aSB}
end

@testset "simulate_bifurcation!" begin
    g = smallgraph(:petersen)
    sys = SimulatedBifurcation{:aSB}(g, randn(ne(g)))
    state = SimulatedBifurcationState(randn(nv(g)), randn(nv(g)))
    simulate_bifurcation!(state, sys; nsteps=100, dt=0.01, clamp=true)
    @test length(state.x) == nv(g)
    @test length(state.p) == nv(g)
    @test all(x -> x >= -1 && x <= 1, state.x)
end
