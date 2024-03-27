struct CSCMatrix{Tv,Ti} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Tv}
    function CSCMatrix(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}) where {Tv, Ti}
        @assert length(colptr) == n + 1
        @assert length(rowval) == length(nzval) == colptr[end] - 1
        new{Tv, Ti}(m, n, colptr, rowval, nzval)
    end
end
Base.size(A::CSCMatrix) = (A.m, A.n)
Base.size(A::CSCMatrix, i::Int) = getindex((A.m, A.n), i)
# the number of non-zero elements
nnz(csc::CSCMatrix) = length(csc.nzval)

function CSCMatrix(coo::COOMatrix)
    m, n = size(coo)
    # sort the COO matrix by column
    order = sortperm(1:nnz(coo); by=i->coo.rowval[i] + m * (coo.colval[i]-1))
    colval, rowval, nzval = coo.colval[order], coo.rowval[order], coo.nzval[order]
    colptr = ones(Int, n+1)
    ptr = 1
    for j in 1:n
        while ptr <= length(colval) && colval[ptr] == j
            ptr += 1
        end
        colptr[j+1] = ptr
    end
    return CSCMatrix(m, n, colptr, rowval, nzval)
end

function Base.getindex(A::CSCMatrix{T}, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    for k in nzrange(A, j)
        if A.rowval[k] == i
            return A.nzval[k]
        end
    end
    return zero(T)
end

function Base.:*(A::CSCMatrix{T1}, B::CSCMatrix{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    @assert size(A, 2) == size(B, 1)
    rowval, colval, nzval = Int[], Int[], T[]
    for j2 in 1:size(B, 2)  # enumerate the columns of B
        for k2 in nzrange(B, j2)  # enumerate the rows of B
            v2 = B.nzval[k2]
            for k1 in nzrange(A, B.rowval[k2])  # enumerate the rows of A
                push!(rowval, A.rowval[k1])
                push!(colval, j2)
                push!(nzval, A.nzval[k1] * v2)
            end
        end
    end
    return CSCMatrix(COOMatrix(size(A, 1), size(B, 2), colval, rowval, nzval))
end

# return the range of non-zero elements in the j-th column
nzrange(A::CSCMatrix, j::Int) = A.colptr[j]:A.colptr[j+1]-1