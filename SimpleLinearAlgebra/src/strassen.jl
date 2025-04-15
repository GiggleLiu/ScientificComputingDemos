function strassen(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    n = size(A, 1)
    n == 1 && return A * B
    @assert iseven(n) && size(A) == size(B) "matrix sizes must be even and equal"

    m = div(n, 2)
    A11, A12 = A[1:m, 1:m], A[1:m, m+1:n]
    A21, A22 = A[m+1:n, 1:m], A[m+1:n, m+1:n]
    B11, B12 = B[1:m, 1:m], B[1:m, m+1:n]
    B21, B22 = B[m+1:n, 1:m], B[m+1:n, m+1:n]

    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = similar(A)
    C[1:m, 1:m] = C11
    C[1:m, m+1:n] = C12
    C[m+1:n, 1:m] = C21
    C[m+1:n, m+1:n] = C22

    return C
end