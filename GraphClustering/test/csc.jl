using Test, GraphClustering.SimpleSparseArrays

@testset "coo2csc" begin
    stiffmatrix = COOMatrix(3, 3, [1, 2, 1, 2, 3, 2, 3], [1, 1, 2, 2, 2, 3, 3], [-1.0, 1, 1, -2, 1, 1, -1])
    csc_matrix = CSCMatrix(stiffmatrix)
    @test Matrix(csc_matrix) ≈ Matrix(stiffmatrix)
end

@testset "csc matmul" begin
    csc_matrix = CSCMatrix(6, 6, [1, 3, 6, 6, 7, 7, 8], [1, 2, 1, 2, 3, 2, 3], [-1.0, 1, 1, -2, 1, 1, -1])
    @test Matrix(csc_matrix)^2 ≈ csc_matrix * csc_matrix
end

@testset "repeated entries" begin
    coo_matrix = COOMatrix(5, 4, [2, 3, 1, 4, 3, 4], [1, 1, 2, 2, 4, 4], [1, 2, 3, 4, 5, 6])
    csc_matrix = CSCMatrix(coo_matrix)
    csc_matrix2 = CSCMatrix(COOMatrix(coo_matrix.n, coo_matrix.m, coo_matrix.rowval, coo_matrix.colval, coo_matrix.nzval))  # transpose
    @test Matrix(csc_matrix) * Matrix(csc_matrix2) ≈ csc_matrix * csc_matrix2
end