struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

mpow2(a::AbstractArray) = a .^ 2

Zygote.@adjoint function LinearAlgebra.svd(A)
    res = LinearAlgebra.svd(A)
    res, function (dy)
        dU, dS, dVt = dy
        return (svd_back(res.U, res.S, res.V, dU, dS, dVt === nothing ? nothing : dVt'),)
    end
end

"""
    svd_back(U, S, V, dU, dS, dV)

Adjoint for SVD decomposition.

References: https://arxiv.org/abs/1909.02659
TODO: try the numerical more stable approach
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    S2 = mpow2(S)
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (mpow2(F) .+ η)

    res = similar(U, (size(S, 1), size(S, 1)))
    fill!(res, 0)
    if !(dU isa Nothing)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    grad = U * res * V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        grad += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        grad += U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    return grad
end