function simplex1d(f, x1, x2; tol=1e-6)
	# initial simplex
	history = [[x1, x2]]
	f1, f2 = f(x1), f(x2)
	while abs(x2 - x1) > tol
		xc = 2x1 - x2
		fc = f(xc)
		if fc < f1   # flip
			x1, f1, x2, f2 = xc, fc, x1, f1
		else         # shrink
			if fc < f2   # let the smaller one be x2.
				x2, f2 = xc, fc
			end
			xd = (x1 + x2) / 2
			fd = f(xd)
			if fd < f1   # update x1 and x2
				x1, f1, x2, f2 = xd, fd, x1, f1
			else
				x2, f2 = xd, fd
			end
		end
		push!(history, [x1, x2])
	end
	return x1, f1, history
end

function simplex(f, x0; tol=1e-6, maxiter=1000)
    n = length(x0)
    x = zeros(n+1, n)
    fvals = zeros(n+1)
    x[1,:] = x0
    fvals[1] = f(x0)
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    for i in 1:n
        x[i+1,:] = x[i,:]
        x[i+1,i] += 1.0
        fvals[i+1] = f(x[i+1,:])
    end
	history = [x]
    for iter in 1:maxiter
        # Sort the vertices by function value
        order = sortperm(fvals)
        x = x[order,:]
        fvals = fvals[order]
        # Calculate the centroid of the n best vertices
        xbar = dropdims(sum(x[1:n,:], dims=1) ./ n, dims=1)
        # Reflection
        xr = xbar + alpha*(xbar - x[n+1,:])
        fr = f(xr)
        if fr < fvals[1]
            # Expansion
            xe = xbar + gamma*(xr - xbar)
            fe = f(xe)
            if fe < fr
                x[n+1,:] = xe
                fvals[n+1] = fe
            else
                x[n+1,:] = xr
                fvals[n+1] = fr
            end
        elseif fr < fvals[n]
            x[n+1,:] = xr
            fvals[n+1] = fr
        else
            # Contraction
            if fr < fvals[n+1]
                xc = xbar + beta*(x[n+1,:] - xbar)
                fc = f(xc)
                if fc < fr
                    x[n+1,:] = xc
                    fvals[n+1] = fc
                else
                    # Shrink
                    for i in 2:n+1
                        x[i,:] = x[1,:] + beta*(x[i,:] - x[1,:])
                        fvals[i] = f(x[i,:])
                    end
                end
            else
                # Shrink
                for i in 2:n+1
                    x[i,:] = x[1,:] + beta*(x[i,:] - x[1,:])
                    fvals[i] = f(x[i,:])
                end
            end
        end
		push!(history, x)
        # Check for convergence
        if maximum(abs.(x[2:end,:] .- x[1,:])) < tol && maximum(abs.(fvals[2:end] .- fvals[1])) < tol
            break
        end
    end
    # Return the best vertex and function value
    bestx = x[1,:]
    bestf = fvals[1]
    return (bestx, bestf, history)
end
