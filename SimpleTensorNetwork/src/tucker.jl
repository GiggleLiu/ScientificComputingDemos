function tucker_movefirst(X::AbstractArray{T, N}, Us, k::Int) where {N, T}
    Ak = X
    for i=1:N
        # move i-th dimension to the first
        if i!=1
            pm = collect(1:N)
            pm[1], pm[i] = pm[i], pm[1]
            Ak = permutedims(Ak, pm)
        end
        if i != k
            # multiply Uk on the i-th dimension
            remain = size(Ak)[2:end]
            Ak = Us[i]' * reshape(Ak, size(Ak, 1), :)
            Ak = reshape(Ak, size(Ak, 1), remain...)
        end
    end
    A_ = permutedims(Ak, (2:N..., 1))
    return permutedims(A_, (k, setdiff(1:N, k)...))
end
function tucker_project(X::AbstractArray{T, N}, Us; inverse=false) where {N, T}
    Ak = X
    for i=1:N
        # move i-th dimension to the first
        if i!=1
            pm = collect(1:N)
            pm[1], pm[i] = pm[i], pm[1]
            Ak = permutedims(Ak, pm)
        end
        remain = size(Ak)[2:end]
        Ak = (inverse ? Us[i] : Us[i]') * reshape(Ak, size(Ak, 1), :)
        Ak = reshape(Ak, size(Ak, 1), remain...)
    end
    return permutedims(Ak, (2:N..., 1))
end

function tucker_decomp(X::AbstractArray{T,N}, rs::Vector{Int}; nrepeat::Int) where {T, N}
    # the first sweep, to generate U_k
    Us = [Matrix{T}(I, size(X, i), size(X, i)) for i=1:N]
    Ak = X
    for n=1:nrepeat
        for i=1:N
            Ak = tucker_movefirst(X, Us, i)
            ret = svd(reshape(Ak, size(Ak, 1), :))
            Us[i] = ret.U[:,1:rs[i]]
        end
        Ak = permutedims(Ak, (2:N..., 1))
        dist = norm(tucker_project(tucker_project(X, Us), Us; inverse=true) .- X)
        @info "The Frobenius norm distance is: $dist"
    end
    return tucker_project(X, Us), Us
end

# X = randn(20, 10, 15);

# Cor, Us = tucker_decomp(X, [4, 5, 6]; nrepeat=10)
