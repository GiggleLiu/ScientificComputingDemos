using MyFirstPackage, Test, MyFirstPackage.CUDA; CUDA.allowscalar(false)

@testset "step!" begin
    lb0 = example_d2q9(; u0=Point(0.0, 0.1))
    lb = deepcopy(lb0)
    for i=1:100 step!(lb) end
    lbc = CUDA.cu(lb0)
    for i=1:100 step!(lbc) end
    # the conservation of mass
    @test all(lb.grid .≈ Array(lbc.grid))
end
