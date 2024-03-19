using Test, MyFirstPackage

@testset "momentum" begin
    lb = D2Q9()
    ds = equilibrium_density(lb, 2.0, Point(0.1, 0.0))
    # the conservation of momentum
    @test momentum(lb, ds) ≈ Point(0.1, 0.0)
    # the conservation of mass
    @test density(ds) ≈ 2.0
end

@testset "step!" begin
    lb0 = example_d2q9(; u0=Point(0.0, 0.1))
    lb = deepcopy(lb0)
    for i=1:100 step!(lb) end
    # the conservation of mass
    @test isapprox(sum(density.(lb.grid)), sum(density.(lb0.grid)); rtol=1e-4)
    # the conservation of momentum
    mean_u = sum(momentum.(Ref(lb.config), lb.grid))/length(lb.grid)
    @test mean_u[2] < 0.1 - 1e-3
end
