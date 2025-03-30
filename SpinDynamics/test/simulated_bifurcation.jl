using SpinDynamics, Test, Graphs

@testset "SimulatedBifurcation" begin
    g = smallgraph(:petersen)
    for KIND in (:aSB, :bSB, :dSB)
        sys = SimulatedBifurcation{KIND}(1.0, 0.2, g, randn(ne(g)))
        x = randn(nv(g))
        f = SpinDynamics.force(sys, x)
        δx = randn(nv(g)) * 1e-5
        engdiff = SpinDynamics.potential_energy(sys, x + δx/2) - SpinDynamics.potential_energy(sys, x - δx/2)
        @test isapprox(sum(f .* δx), engdiff, rtol=1e-3)
    end
end