module GraphClustering

using LinearAlgebra
using KrylovKit
using Graphs, LuxorGraphPlot
using Clustering

export SimpleSparseArrays, GraphViz
export setcolor!, setlabel!, setsize!, drawing
export glue_graphs, spectral_clustering

include("clustering.jl")
include("visualization.jl")

# NOTE: This module is only for tutoring, you should use the standard library `SparseArrays` in your project
module SimpleSparseArrays

export COOMatrix, CSCMatrix

include("coo.jl")
include("csc.jl")
end

end
