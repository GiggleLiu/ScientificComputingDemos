module HiddenMarkovModel

using OMEinsum
using LinearAlgebra
using Random, StatsBase

export HMM, forward, backward, viterbi, baum_welch, generate_sequence

include("hmm.jl")

end
