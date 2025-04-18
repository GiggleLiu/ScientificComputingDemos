# NOTE: it's simpler to just manually define the overloads of ChainRules
# so let's not generate them automatically for now, until we can handle
# the function signatures better, e.g wrap tracked-types over supported
# types of the rrule signatures automatically
Base.reshape(A::TrackedArray, shape::Int...) = trace(reshape, A, shape...)
Base.reshape(A::TrackedArray, shape::NTuple{N, Int}) where N = trace(reshape, A, shape)
Base.vect(X::Vararg{<:TrackedArray}) = trace(Base.vect, X)
Base.hcat(Xs::TrackedArray...) = trace(Base.hcat, Xs...)

Base.:(+)(A::TrackedArray, B::TrackedArray) = trace(+, A, B)
Base.:(-)(A::TrackedArray, B::TrackedArray) = trace(-, A, B)
Base.:(*)(A::TrackedArray, B::TrackedArray) = trace(*, A, B)
Base.:(-)(A::TrackedArray) = trace(-, A)

function Base.sum(A::TrackedArray)
    ret = fill!(similar(A.value, ()), sum(A.value))
    A̅ = zero(A)
    record = Record(sum, (A,), y̅ -> (ChainRules.NoTangent(), fill!(A̅, y̅[])), zero(ret), false)
    return TrackedArray(ret, record)
end
function Base.getindex(A::TrackedArray, indices::Int...)
    ret = fill!(similar(A.value, ()), A.value[indices...])
    A̅ = zero(A)
    A̅[indices...] = 1
    record = Record(getindex, (A, indices), y̅ -> (ChainRules.NoTangent(), A̅), zero(ret), false)
    return TrackedArray(ret, record)
end

function Base.abs2(A::TrackedArray)
    @assert ndims(A) == 0 "expect a scalar input for abs2"
    ret = map(abs2, A.value)
    A̅ = zero(A.value)
    record = Record(abs2, (A,), y̅ -> (ChainRules.NoTangent(), map((x, y)-> 2*x*y, A.value, y̅)), zero(ret), false)
    return TrackedArray(ret, record)
end
