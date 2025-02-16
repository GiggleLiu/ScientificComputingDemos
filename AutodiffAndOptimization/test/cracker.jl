using Test
using AutodiffAndOptimization.Cracker
using AutodiffAndOptimization.Cracker.ChainRules: rrule, unthunk

@testset "basic propagate ($T)" for T in [Float64, ComplexF64]
    A, B, C = track(rand(T, 2, 2)), track(rand(T, 2, 2)), track(rand(T, 4))
    Z = A + B * reshape(C, 2, 2)
    
    ret = abs2(sum(Z))
    Cracker.backpropagate!(ret, fill!(similar(ret), 1.0))
    
    uA,uB,uC = untrack.((A, B, C))
    T1, pb1 = rrule(reshape, uC, 2, 2)
    T2, pb2 = rrule(*, uB, T1)
    T3, pb3 = rrule(+, uA, T2)
    T4, pb4 = rrule(sum, T3)
    T5, pb5 = rrule(abs2, T4)
    
    dT5 = 1.0
    _, dT4 = pb5(dT5)
    _, dT3 = pb4(dT4)
    _, duA, dT2 = pb3(dT3)
    _, duB, dT1 = pb2(dT2)
    _, duC, _, _ = pb1(dT1)
    
    @test unthunk(duA) ≈ A.record.grad
    @test unthunk(duB) ≈ B.record.grad
    @test unthunk(duC) ≈ C.record.grad
end

@testset "gradient 1" begin
    a = rand(2, 2)
    @test Cracker.gradient(sum, (a,))[1] == ones(2, 2)
end
    
@testset "gradient 2" begin
    A, B, C = rand(Float64, 2, 2), rand(Float64, 2, 2), rand(Float64, 4)
    function loss(A, B, C)
        Z = A + B * reshape(C, 2, 2)
        return abs2(sum(Z))
    end
    grads = Cracker.gradient(loss, (A, B, C))
    @test grads[1] ≈ ones(2, 2)
    @test grads[2] ≈ ones(2, 2)
    @test grads[3] ≈ ones(2, 2)
end