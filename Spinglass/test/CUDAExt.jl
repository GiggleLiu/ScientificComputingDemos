using Spinglass, Test, CUDA

@testset "anneal" begin
    sap = load_spinglass(pkgdir(Spinglass, "data", "example.txt"))
    tempscales = 10 .- (1:64 .- 1) .* 0.15 |> collect
    cusap = SpinGlassSA(sap) |> CUDA.cu
    opt_cost, opt_config = anneal(30, cusap, CUDA.CuVector(tempscales), 4000)
    @test opt_cost == -3858
end