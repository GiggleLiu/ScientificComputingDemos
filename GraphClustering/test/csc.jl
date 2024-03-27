using Test, GraphClustering

@testset "coo2csc" begin
    stiffmatrix = COOMatrix(3, 3, [1, 2, 1, 2, 3, 2, 3], [1, 1, 2, 2, 2, 3, 3], [-1.0, 1, 1, -2, 1, 1, -1])
    csc_matrix = CSCMatrix(stiffmatrix)
    @test Matrix(csc_matrix) ≈ Matrix(stiffmatrix)
end

@testset "csc matmul" begin
    csc_matrix = CSCMatrix(6, 6, [1, 1, 2, 2, 2, 5, 8], [1, 2, 1, 2, 3, 2, 3], [-1.0, 1, 1, -2, 1, 1, -1])
    @test Matrix(csc_matrix)^2 ≈ csc_matrix * csc_matrix
end
