using Test
using Zygote, OMEinsum
using TensorRenormalizationGroup: trg, trg_svd

@testset "unit" begin
    t = randn(10,10,10,10)
    u, v = trg_svd(t, 100, 0)
    @test t ≈ ein"ija,akl -> ijkl"(u,v)
end

@testset "real" begin
    χ, niter = 5, 5
    foo = β -> trg(model_tensor(Ising(),β), χ, niter).lnZ
    # the pytorch result with tensorgrad
    # https://github.com/wangleiphy/tensorgrad
    # clone this repo and type
    # $ python 1_ising_TRG/ising.py -chi 5 -Niter 5
    @test foo(0.4) ≈ 0.8919788686747141
end