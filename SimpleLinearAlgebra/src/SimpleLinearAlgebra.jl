module SimpleLinearAlgebra
# import packages
using LinearAlgebra
import FFTW

# export interfaces
export back_substitution!
export elementary_elimination_matrix
export lufact_naive!, lufact!
export lufact_pivot!
export HouseholderMatrix, left_mul!, right_mul!, householder_e1, householder_qr!
export qr_left_mul!, qr_right_mul!, givens_matrix, givens_qr!
export classical_gram_schmidt, modified_gram_schmidt!
export dft_matrix
export fft!, ifft!

# `include` other source files into this module
include("back_substitution.jl")
include("lu_factorization.jl")
include("lu_factorization_partialpivoting.jl")
include("householder.jl")
include("qr_factorization.jl")
include("orthogonalization.jl")
include("fouriertransform.jl")
include("fastfouriertransform.jl")

end # module

