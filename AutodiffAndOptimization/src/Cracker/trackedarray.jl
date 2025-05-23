mutable struct Record{T <: AbstractArray}
    f
    args
    pullback
    grad::T
    is_leaf::Bool
end

leaf_record(x) = Record(nothing, nothing, nothing, zero(x), true)
# we use traits over types
is_tracked(x) = false
untrack(x) = is_tracked(x) ? x.value : x
record(x) = x.record

struct TrackedArray{T, N, S <: AbstractArray{T, N}} <: AbstractArray{T, N}
    value::S
    record::Record{S}
end

function Base.show(io::IO, mime::MIME"text/plain", x::TrackedArray)
    print(io, "tracked ")
    show(io, mime, x.value)
end

track(A::AbstractArray, record::Record=leaf_record(A)) = TrackedArray(A, record)
is_tracked(::TrackedArray) = true
Base.IndexStyle(X::TrackedArray) = IndexStyle(untrack(X))
Base.size(X::TrackedArray, idx::Int...) = size(untrack(X), idx...)
Base.length(X::TrackedArray) = length(untrack(X))

is_tracked(x::Tuple) = any(is_tracked, x)
untrack(x::Tuple) = untrack.(x)
function track(A::Tuple, record::Record=leaf_record(A))
    return track.(A, Ref(record))
end

function Base.show(io::IO, x::TrackedArray)
    print(io, "track(")
    show(io, x.value)
    print(io, ")")
end