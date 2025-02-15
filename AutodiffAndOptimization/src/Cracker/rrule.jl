# NOTE: it's simpler to just manually define the overloads of ChainRules
# so let's not generate them automatically for now, until we can handle
# the function signatures better, e.g wrap tracked-types over supported
# types of the rrule signatures automatically
Base.getindex(A::TrackedArrayType, indices::Int...) = trace(getindex, A, indices...)
Base.reshape(A::TrackedArrayType, shape::Int...) = trace(reshape, A, shape...)
Base.reshape(A::TrackedArrayType, shape::NTuple{N, Int}) where N = trace(reshape, A, shape)
Base.vect(X::Vararg{<:TrackedArrayType}) = trace(Base.vect, X)
Base.hcat(Xs::TrackedArrayType...) = trace(Base.hcat, Xs...)

Base.:(+)(A::TrackedType, B::TrackedType) = trace(+, A, B)
Base.:(-)(A::TrackedType, B::TrackedType) = trace(-, A, B)
Base.:(*)(A::TrackedType, B::TrackedType) = trace(*, A, B)
Base.:(-)(A::TrackedType) = trace(-, A)

Base.prod(A::TrackedArrayType) = trace(prod, A)
Base.sum(A::TrackedArrayType) = trace(sum, A)
Base.abs(X::TrackedType) = trace(abs, X)
Base.abs2(X::TrackedType) = trace(abs2, X)
