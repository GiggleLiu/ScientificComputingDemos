using Test
@testset "pflip" begin
    model = IsingModel(10, 0.1, 0.5)
    @test isapprox(pflip(model, -1, -4), 0.0202419; rtol=1e-4)
    @test isapprox(pflip(model, 1, -4), 49.4024; rtol=1e-4)
    @test isapprox(pflip(model, -1, -2), 0.149569; rtol=1e-4)
    @test isapprox(pflip(model, 1, -2), 6.68589; rtol=1e-4)
    @test isapprox(pflip(model, -1, 0), 1.10517; rtol=1e-4)
    @test isapprox(pflip(model, 1, 0), 0.904837; rtol=1e-4)
    @test isapprox(pflip(model, -1, 2), 8.16617; rtol=1e-4)
    @test isapprox(pflip(model, 1, 2), 0.122456; rtol=1e-4)
    @test isapprox(pflip(model, -1, 4), 60.3403; rtol=1e-4)
    @test isapprox(pflip(model, 1, 4), 0.0165727; rtol=1e-4)
end

@testset "energy" begin
    model = IsingModel(10, 0.0, 0.1)
    spin = fill(-1, model.l, model.l)
    @test energy(model, spin) ≈ -200
    model = IsingModel(10, 0.1, 0.0)
    spin = fill(-1, model.l, model.l)
    @test energy(model, spin) ≈ -190
end