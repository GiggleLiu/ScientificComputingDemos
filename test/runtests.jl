using SimpleLinearAlgebra
using Test

@testset "SimpleLinearAlgebra.jl" begin
    include("back_substitution.jl")
    include("lu_factorization.jl")
    include("lu_factorization_partialpivoting.jl")
    include("householder.jl")
    include("qr_factorization.jl")
    include("orthogonalization.jl")
end

