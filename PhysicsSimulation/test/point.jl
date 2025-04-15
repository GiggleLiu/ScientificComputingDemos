using Test, PhysicsSimulation, LinearAlgebra

@testset "Point Construction" begin
    # Basic construction
    p = Point(1.0, 2.0)
    @test p isa Point{2, Float64}
    @test p.data == (1.0, 2.0)

    # Type specific construction
    p_int = Point(1, 2)
    @test p_int isa Point{2, Int}

    # Convenience types
    p2d = Point2D{Float64}((1.0, 2.0))
    p3d = Point3D{Float64}((1.0, 2.0, 3.0))
    @test p2d isa Point2D{Float64}
    @test p3d isa Point3D{Float64}
end

@testset "Point Arithmetic" begin
    p1 = Point(1.0, 2.0)
    p2 = Point(3.0, 4.0)
    
    # Addition
    @test p1 + p2 == Point(4.0, 6.0)
    
    # Subtraction
    @test p2 - p1 == Point(2.0, 2.0)
    
    # Scalar multiplication
    @test 2 * p1 == Point(2.0, 4.0)
    @test p1 * 2 == Point(2.0, 4.0)
    
    # Division
    @test p1 / 2 == Point(0.5, 1.0)
end

@testset "Point Operations" begin
    p1 = Point(1.0, 2.0)
    p2 = Point(3.0, 4.0)
    
    # Dot product
    @test dot(p1, p2) ≈ 11.0
    
    # Distance
    @test PhysicsSimulation.distance(p1, p2) ≈ sqrt(8.0)
    
    # Zero
    @test zero(Point{2, Float64}) == Point(0.0, 0.0)
    @test zero(p1) == Point(0.0, 0.0)
end

@testset "Point Utilities" begin
    p = Point(1.0, 2.0)
    @test length(p) == 2
    
    # Indexing
    @test p[1] == 1.0
    @test p[2] == 2.0
    
    # Iteration
    collected = collect(p)
    @test collected == [1.0, 2.0]
    
    # Approximate equality
    p1 = Point(1.0, 2.0)
    p2 = Point(1.0 + 1e-10, 2.0 - 1e-10)
    @test isapprox(p1, p2, atol=1e-9)
end
