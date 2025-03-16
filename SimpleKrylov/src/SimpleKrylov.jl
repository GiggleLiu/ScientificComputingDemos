module SimpleKrylov

export lanczos_reorthogonalize
export arnoldi_iteration

# NOTE: This module is only for tutoring, you should use the standard library `SparseArrays` in your project
module SimpleSparseArrays

export COOMatrix, CSCMatrix

include("coo.jl")
include("csc.jl")

end

using .SimpleSparseArrays
using LinearAlgebra

include("lanczos.jl")
include("arnoldi.jl")

end
