module Cracker

using ChainRules: rrule, unthunk
import ChainRules
using LinearAlgebra

export track, untrack

include("types/record.jl")
include("types/number.jl")
include("types/array.jl")

is_tracked(x::Tuple) = any(is_tracked, x)
untrack(x::Tuple) = untrack.(x)
function track(A::Tuple, record::Record=leaf(A))
    return track.(A, Ref(record))
end

const TrackedType = Union{TrackedNumber, TrackedArrayType}
function Base.show(io::IO, x::TrackedType)
    print(io, "track(")
    show(io, x.value)
    print(io, ")")
end


include("trace.jl")
include("rrule.jl")

end
