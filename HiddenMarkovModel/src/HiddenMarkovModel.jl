module HiddenMarkovModel

using OMEinsum
using LinearAlgebra
using StatsBase

export HMM, viterbi, baum_welch, generate_sequence
export HMMNetwork, likelihood, likelihood_and_gradient

include("hmm.jl")

end
