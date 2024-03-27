struct COOMatrix{Tv, Ti} <: AbstractArray{Tv, 2}   # Julia does not have a COO data type
    m::Ti                   # number of rows
    n::Ti                 # number of columns
    colval::Vector{Ti}   # column indices
    rowval::Vector{Ti}   # row indices
    nzval::Vector{Tv}        # values
    function COOMatrix(m::Ti, n::Ti, colval::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv, Ti}
        @assert length(colval) == length(rowval) == length(nzval)
        new{Tv, Ti}(m, n, colval, rowval, nzval)
    end
end

Base.size(coo::COOMatrix) = (coo.m, coo.n)
Base.size(coo::COOMatrix, i::Int) = getindex((coo.m, coo.n), i)
# the number of non-zero elements
nnz(coo::COOMatrix) = length(coo.nzval)

function Base.getindex(coo::COOMatrix{Tv}, i::Integer, j::Integer) where Tv
    @boundscheck checkbounds(coo, i, j)
    v = zero(Tv)
    for (i2, j2, v2) in zip(coo.rowval, coo.colval, coo.nzval)
        if i == i2 && j == j2
            v += v2  # accumulate the value, since repeated indices are allowed.
        end
    end
    return v
end

function Base.:(*)(A::COOMatrix{T1}, B::COOMatrix{T2}) where {T1, T2}
    @assert size(A, 2) == size(B, 1)
    rowval = Int[]
    colval = Int[]
    nzval = promote_type(T1, T2)[]
    for (i, j, v) in zip(A.rowval, A.colval, A.nzval)
        for (i2, j2, v2) in zip(B.rowval, B.colval, B.nzval)
            if j == i2
                push!(rowval, i)
                push!(colval, j2)
                push!(nzval, v * v2)
            end
        end
    end
    return COOMatrix(size(A, 1), size(B, 2), colval, rowval, nzval)
end