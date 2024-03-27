using Test, GraphClustering.SimpleSparseArrays

@testset "coo matmul" begin
    stiffmatrix = COOMatrix(3, 3, [1, 2, 1, 2, 3, 2, 3], [1, 1, 2, 2, 2, 3, 3], [-1.0, 1, 1, -2, 1, 1, -1])
    @test stiffmatrix[2, 3] == 1
    dense_matrix = Matrix(stiffmatrix)
    @test stiffmatrix * stiffmatrix ≈ dense_matrix ^ 2
end