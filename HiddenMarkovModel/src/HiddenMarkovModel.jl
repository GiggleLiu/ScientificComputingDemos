module HiddenMarkovModel

using OMEinsum
using LinearAlgebra
using StatsBase

export HMM, forward, backward, viterbi, baum_welch, generate_sequence
export HMMNetwork

include("hmm.jl")

end
