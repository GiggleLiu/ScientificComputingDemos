using Test, SimpleTensorNetwork, SimpleTensorNetwork.Graphs, SimpleTensorNetwork.OMEinsum

@testset "spinglass" begin
    sg = Spinglass(smallgraph(:petersen), ones(15), zeros(10))
    β = 0.1
    tn = generate_tensor_network(sg, β)
    @test length(tn.tensors) == 25
    opttn = optimize_tensornetwork(tn)
    @test length(opttn.tensors) == 25
    @test opttn.ein isa OMEinsum.SlicedEinsum
    @test partition_function(sg, β) ≈ partition_function_exact(sg, β)
end