module Cracker

using ChainRules: rrule, unthunk
import ChainRules
using LinearAlgebra

export track, untrack

include("trackedarray.jl")
include("trace.jl")
include("rrule.jl")

end
