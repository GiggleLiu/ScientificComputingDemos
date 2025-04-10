using Spinglass, Test, Random, Graphs

@testset "loading" begin
    sap = load_spinglass(pkgdir(Spinglass, "data", "example.txt"))
    @test nv(sap.graph) == 300
    sapsa = Spinglass.SpinGlassSA(sap)
    @test size(sapsa.coupling) == (300, 300)
end

@testset "random config" begin
    sap = Spinglass.SpinGlassSA(load_spinglass(pkgdir(Spinglass, "data", "example.txt")))
    initial_config = random_config(sap)
    @test initial_config.config |> length == 300
    @test eltype(initial_config.config) == Int
end

@testset "anneal" begin
    sap = load_spinglass(pkgdir(Spinglass, "data", "example.txt"))
    tempscales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    opt_cost, opt_config = anneal(30, sap, tempscales, 4000)
    @test anneal(30, sap, tempscales, 4000)[1] == -3858
end