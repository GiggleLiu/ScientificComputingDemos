module GraphClustering

using LinearAlgebra
using KrylovKit
using Graphs  # for generating sparse matrices

export SimpleSparseArrays

# NOTE: This module is only for tutoring, you should use the standard library `SparseArrays` in your project
module SimpleSparseArrays

export COOMatrix, CSCMatrix

include("coo.jl")
include("csc.jl")
end

end
