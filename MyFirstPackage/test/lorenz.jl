using Test, MyFirstPackage

@testset "Point" begin
    p1 = Point(1.0, 2.0)
    p2 = Point(3.0, 4.0)
    @test p1 + p2 ≈ Point(4.0, 6.0)
end

@testset "step" begin
    lz = Lorenz(10.0, 28.0, 8/3)
    int = RungeKutta{4}()
    r1 = integrate_step(lz, int, Point(1.0, 1.0, 1.0), 0.0001)
    eu = Euclidean()
    r2 = integrate_step(lz, eu, Point(1.0, 1.0, 1.0), 0.0001)
    @test isapprox(r1, r2; rtol=1e-5)
end