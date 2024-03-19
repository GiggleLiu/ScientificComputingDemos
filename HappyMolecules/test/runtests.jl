using HappyMolecules
using Test, Random

using StaticArrays
using HappyMolecules.Applications: lennard_jones_triple_point

@testset "location initialization" begin
    Random.seed!(2)
    # random locations in an integer box size
    box = PeriodicBox(10, 10)
    @test HappyMolecules.largest_distance(box) == 5 * sqrt(2)
    locs = random_locations(box, 10000)
    @test length(locs) == 10000
    @test isapprox(sum(locs)/10000, SVector(4.5, 4.5); atol=5e-2)
    @test HappyMolecules.volume(box) == 100
    
    # random locations in a floating point box size
    box = PeriodicBox(10.0, 10.0)
    locs = random_locations(box, 10000)
    @test length(locs) == 10000
    @test isapprox(sum(locs)/10000, SVector(5.0, 5.0); atol=5e-2)

    # uniform locations in a floating point box size
    box = PeriodicBox(10.0, 10.0)
    locs = uniform_locations(box, 4)
    @test length(locs) == 4
    @test locs ≈ [SVector(0.0, 0.0), SVector(5.0, 0.0), SVector(0.0, 5.0), SVector(5.0, 5.0)]
end

@testset "binning" begin
    # constructor
    bin = Bin(-1.0, 1.0, 20)
    # ticks
    @test ticks(bin) ≈ [-1.05 + 0.1 * i for i=1:20]

    # push!
    @test_throws AssertionError push!(bin, -2.0)
    @test_throws AssertionError push!(bin, 2.0)
    r1 = zeros(Int, 20); r1[end] += 1
    @test push!(bin, 0.99).counts == r1
    r1[end] += 1
    @test push!(bin, 0.91).counts == r1
    r1[end-1] += 1
    @test push!(bin, 0.89).counts == r1

    # ncounts
    @test ncounts(bin) == 3

    # empty!
    empty!(bin)
    @test ncounts(bin) == 0
end

@testset "enzyme potential field" begin
    potential, vector = LennardJones(), SVector(1.0, 2.0, 1.0)
    ef = HappyMolecules.enzyme_potential_field(potential, vector)
    field = force(potential, vector)
    @test field ≈ ef
end

@testset "LennardJones potential" begin
    rc = 2.519394287073761
    rc2 = rc ^ 2
    ecut = 4 * (1/rc2^6 - 1/rc2^3)
    p = LennardJones(; rc)
    @test p.ecut ≈ ecut
    @test isapprox(potential_energy(p, SVector(0.0, rc)), 0; atol=1e-8)
end

@testset "LennardJones triple point" begin
    res = lennard_jones_triple_point()
    md = res.runtime
    @test isapprox(HappyMolecules.temperature(md), 1.4595; atol=0.01)
    @test isapprox(HappyMolecules.pressure(md), 5.27; atol=5e-2)
    # energy conservation
    ks, ps = res.kinetic_energy, res.potential_energy
    @test isapprox(ps[1] + ks[1], ps[end] + ks[end]; atol=1e-2)
end