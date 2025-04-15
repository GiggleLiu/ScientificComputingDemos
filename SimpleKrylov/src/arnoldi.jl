function arnoldi_iteration(A::AbstractMatrix{T}, x0::AbstractVector{T}; maxiter) where T
    # Storage for Hessenberg matrix entries (column by column)
    h = Vector{T}[]
    # Storage for orthonormal basis vectors of the Krylov subspace
    q = [normalize(x0)]
    n = length(x0)
    # Ensure A is a square matrix of appropriate dimensions
    @assert size(A) == (n, n)
    
    # Main Arnoldi iteration loop
    for k = 1:min(maxiter, n)
        # Apply the matrix to the latest basis vector
        u = A * q[k]    # generate next vector
        
        # Initialize the k-th column of the Hessenberg matrix
        hk = zeros(T, k+1)
        
        # Orthogonalize against all previous basis vectors (Gram-Schmidt process)
        for j = 1:k # subtract from new vector its components in all preceding vectors
            hk[j] = q[j]' * u  # Calculate projection coefficient
            u = u - hk[j] * q[j]  # Subtract projection
        end
        
        # Calculate the norm of the remaining vector
        hkk = norm(u)
        hk[k+1] = hkk  # This will be the subdiagonal entry
        push!(h, hk)  # Store this column of coefficients
        
        # Check for convergence or breakdown
        if abs(hkk) < 1e-8 || k >= n # stop if matrix is reducible
            break
        else
            # Normalize the new basis vector and add to our collection
            push!(q, u ./ hkk)
        end
    end

    # Construct the Hessenberg matrix H from the stored coefficients
    kmax = length(h)
    H = zeros(T, kmax, kmax)
    for k = 1:length(h)
        if k == kmax
            # Last column might be shorter if we had early termination
            H[1:k, k] .= h[k][1:k]
        else
            # Standard case: copy the full column including subdiagonal entry
            H[1:k+1, k] .= h[k]
        end
    end
    
    # Return the Hessenberg matrix and the orthonormal basis matrix
    return H, hcat(q...)
end
