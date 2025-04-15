function lanczos(A, q1::AbstractVector{T}; abstol, maxiter) where T
    # Normalize the initial vector
    q1 = normalize(q1)
    
    # Initialize storage for basis vectors and tridiagonal matrix elements
    q = [q1]                # Orthonormal basis vectors
    α = [q1' * (A * q1)]    # Diagonal elements of tridiagonal matrix
    
    # Compute first residual: r₁ = Aq₁ - α₁q₁
    Aq1 = A * q1
    rk = Aq1 .- α[1] .* q1
    β = [norm(rk)]          # Off-diagonal elements of tridiagonal matrix
    
    # Main Lanczos iteration
    for k = 2:min(length(q1), maxiter)
        # Compute next basis vector: q_k = r_{k-1}/β_{k-1}
        push!(q, rk ./ β[k-1])
        
        # Compute A*q_k
        Aqk = A * q[k]
        
        # Compute diagonal element: α_k = q_k' * A * q_k
        push!(α, q[k]' * Aqk)
        
        # Compute residual: r_k = A*q_k - α_k*q_k - β_{k-1}*q_{k-1}
        # This enforces orthogonality to the previous two vectors
        rk = Aqk .- α[k] .* q[k] .- β[k-1] * q[k-1]
        
        # Compute the norm of the residual for the off-diagonal element
        nrk = norm(rk)
        
        # Check for convergence or maximum iterations
        if abs(nrk) < abstol || k == length(q1)
            break
        end
        
        push!(β, nrk)
    end
    
    # Return the tridiagonal matrix T and orthogonal matrix Q
    return SymTridiagonal(α, β), hcat(q...)
end

function lanczos_reorthogonalize(A, q1::AbstractVector{T}; abstol, maxiter) where T
    n = length(q1)
    
    # Normalize the initial vector
    q1 = normalize(q1)
    
    # Initialize storage
    q = [q1]                # Orthonormal basis vectors
    α = [q1' * (A * q1)]    # Diagonal elements of tridiagonal matrix
    Aq1 = A * q1
    rk = Aq1 .- α[1] .* q1
    β = [norm(rk)]          # Off-diagonal elements of tridiagonal matrix
    
    # Store Householder transformations for reorthogonalization
    householders = [householder_matrix(q1)]
    
    # Main Lanczos iteration with reorthogonalization
    for k = 2:min(n, maxiter)
        # Step 1: Apply all previous Householder transformations to residual vector
        # This ensures full orthogonality to all previous vectors
        for j = 1:k-1
            left_mul!(view(rk, j:n), householders[j])
        end
        
        # Create new Householder transformation for the current residual
        push!(householders, householder_matrix(view(rk, k:n)))
        
        # Step 2: Compute the k-th orthonormal vector by applying Householder transformations
        # Start with unit vector e_k and apply all Householder transformations in reverse
        qk = zeros(T, n)
        qk[k] = 1  # qₖ = H₁H₂…Hₖeₖ
        for j = k:-1:1
            left_mul!(view(qk, j:n), householders[j])
        end
        push!(q, qk)
        
        # Compute A*q_k
        Aqk = A * q[k]
        
        # Compute diagonal element: α_k = q_k' * A * q_k
        push!(α, q[k]' * Aqk)
        
        # Compute residual: r_k = A*q_k - α_k*q_k - β_{k-1}*q_{k-1}
        rk = Aqk .- α[k] .* q[k] .- β[k-1] * q[k-1]
        
        # Compute the norm of the residual
        nrk = norm(rk)
        
        # Check for convergence or maximum iterations
        if abs(nrk) < abstol || k == n
            break
        end
        
        push!(β, nrk)
    end
    
    # Return the tridiagonal matrix T and orthogonal matrix Q
    return SymTridiagonal(α, β), hcat(q...)
end

# Householder transformation matrix representation
struct HouseholderMatrix{T} <: AbstractArray{T, 2}
    v::Vector{T}    # Householder vector
    β::T            # Scaling factor
end

# Apply Householder transformation: B = (I - β*v*v')*B
function left_mul!(B, A::HouseholderMatrix)
    # Compute v'*B
    vB = A.v' * B
    # Apply transformation: B = B - β*v*(v'*B)
    B .-= (A.β .* A.v) * vB
    return B
end

# Create a Householder matrix that transforms v to a multiple of e₁
function householder_matrix(v::AbstractVector{T}) where T
    v = copy(v)
    # Modify first element to ensure numerical stability
    v[1] -= norm(v, 2)
    # Compute scaling factor β = 2/||v||²
    return HouseholderMatrix(v, 2/norm(v, 2)^2)
end
