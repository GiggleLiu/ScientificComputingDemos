using Test, PhysicsSimulation

@testset "planets" begin
    @test length(solar_system()) == 10
    acc = zeros(Point3D{Float64}, length(solar_system()))
    @test length(PhysicsSimulation.update_acceleration!(acc, solar_system())) == 10
end

@testset "leapfrog" begin
    cached = LeapFrogSystem(solar_system())
    newcache = step!(cached, 0.1)
    @test newcache isa LeapFrogSystem

    res = leapfrog_simulation(solar_system(); dt=0.01, nsteps=55)
    @test res[end].sys.bodies[1].r ≈ PhysicsSimulation.Point(-0.002580393612084354, 0.0008688001295124886, 2.269033380228867e-6)
end