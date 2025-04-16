using TreeverseAlgorithm, CairoMakie, Enzyme, LinearAlgebra

"""
    lorentz(t, p, θ)

Compute the Lorenz system derivatives at time `t` for state `p` with parameters `θ`.

# Arguments
- `t`: Time (not used in autonomous system but included for ODE solver compatibility)
- `p`: State vector [x, y, z]
- `θ`: Parameter tuple (ρ, σ, β)

# Returns
- Vector of derivatives [dx/dt, dy/dt, dz/dt]
"""
function lorentz(t, p, θ)
    ρ, σ, β = θ
    x, y, z = p
    return [σ*(y-x), ρ*x-y-x*z, x*y-β*z]
end

"""
    rk4_step(f, t, y, θ; Δt)

Perform a single 4th-order Runge-Kutta integration step.

# Arguments
- `f`: Function defining the ODE system
- `t`: Current time
- `y`: Current state vector
- `θ`: Parameters for the ODE system
- `Δt`: Time step size

# Returns
- Updated state vector after one step
"""
function rk4_step(f, t, y, θ; Δt)
    k1 = Δt/6 * f(t, y, θ)
    k2 = Δt/3 * f(t+Δt/2, y .+ k1 ./ 2, θ)
    k3 = Δt/3 * f(t+Δt/2, y .+ k2 ./ 2, θ)
    k4 = Δt/6 * f(t+Δt, y .+ k3, θ)
    return y + k1 + k2 + k3 + k4
end

"""
    rk4(f, y0, θ; t0, Δt, Nt)

Integrate an ODE system using the 4th-order Runge-Kutta method.

# Arguments
- `f`: Function defining the ODE system
- `y0`: Initial state vector
- `θ`: Parameters for the ODE system
- `t0`: Initial time
- `Δt`: Time step size
- `Nt`: Number of time steps

# Returns
- Final state vector after integration
"""
function rk4(f, y0::T, θ; t0, Δt, Nt) where T
    for i=1:Nt
        y0 = rk4_step(f, t0+(i-1)*Δt, y0, θ; Δt=Δt)
    end
    return y0
end

"""
    gradient_treeverse(x, θ; Δt, N)

Compute gradients of the Lorenz system using the Treeverse algorithm.

# Arguments
- `x`: Initial state vector
- `θ`: Parameters (ρ, σ, β)
- `Δt`: Time step size
- `N`: Number of steps

# Returns
- Tuple of (gradient, log)
"""
function gradient_treeverse(x, θ; Δt, N)
    step_func(x) = rk4_step(lorentz, 0.0, x, θ; Δt)
    
    function back(x, f_and_g::Nothing)
        # Set the gradient of the last state to [1, 0, 0], i.e., differentiate with respect to x
        y = step_func(x)
        return back(x, (y, y, [1.0, 0.0, 0.0]))
    end
    
    function back(x, f_and_g::Tuple)
        function forward(x, y)
            y .= step_func(x)
            return nothing
        end
        x̅ = zero(x)
        result, y, y̅ = f_and_g
        Enzyme.autodiff(Enzyme.Reverse, Const(forward), Duplicated(x, x̅), Duplicated(y, y̅))
        return (result, x, x̅)
    end
    
    logger = TreeverseAlgorithm.TreeverseLog()
    result_tv, _, g_tv = treeverse(step_func, back, x; δ=40, N, logger)
    return g_tv, logger
end

"""
    create_gradient_heatmap(rhos, sigmas, gradients; filename)

Create and save a heatmap visualization of gradient norms.

# Arguments
- `rhos`: Range of ρ values
- `sigmas`: Range of σ values
- `gradients`: Matrix of gradient norms
- `filename`: Output file path

# Returns
- Figure object
"""
function create_gradient_heatmap(rhos, sigmas, gradients; filename=joinpath(@__DIR__, "gradient_heatmap.png"))
    fig = Figure(size = (800, 600))

    ax = Axis(fig[1, 1], 
              xlabel = "ρ", 
              ylabel = "σ", 
              title = "Gradient Norm Heatmap (Lorenz System)")

    # Create the heatmap with log scale
    hm = heatmap!(ax, rhos, sigmas, gradients', 
                  colormap = :viridis, 
                  colorscale = log10,
                  colorrange = (1e-3, 1e5))

    # Add a colorbar with log scale labels
    Colorbar(fig[1, 2], hm, label = "Gradient Norm (log scale)")

    # Save the figure
    save(filename, fig)
    println("Log-scale heatmap saved to $(filename)")
    
    return fig
end

# Reference: Jin-Guo L., Kai-Lai X., 2021. Automatic differentiation and its applications in physics simulation. 
# Acta Phys. Sin. 70, 149402–11. https://doi.org/10.7498/aps.70.20210813

# Define parameter ranges
rhos = 0:2:50   # Using a step of 2 to reduce computation time
sigmas = 0:2:30 # Using a step of 2 to reduce computation time
gradients = zeros(length(rhos), length(sigmas))

@info "Starting gradient computation for Lorenz system"
@info "Parameter ranges: ρ ∈ [$(minimum(rhos)), $(maximum(rhos))], σ ∈ [$(minimum(sigmas)), $(maximum(sigmas))]"
@info "Total parameter combinations to evaluate: $(length(rhos) * length(sigmas))"

# Compute gradients for each parameter combination
for (i, ρ) in enumerate(rhos)
    for (j, σ) in enumerate(sigmas)
        x0 = [1.0, 0.0, 0.0]
        θ = (ρ, σ, 8/3)  # Standard β value for Lorenz system
        
        @info "Computing gradient for parameters" ρ=ρ σ=σ β=8/3 progress="$((i-1)*length(sigmas)+j)/$(length(rhos)*length(sigmas))"
        g_tv, logger = gradient_treeverse(x0, θ; Δt = 1e-3, N = 10000)
        gradients[i, j] = norm(g_tv)
        
        @info "Gradient computation complete" ρ=ρ σ=σ gradient_norm=norm(g_tv)
        @info logger
    end
end

@info "Gradient computation completed for all parameter combinations"
@info "Generating heatmap visualization"

# Generate and display the heatmap
fig = create_gradient_heatmap(rhos, sigmas, gradients)

@info "Analysis complete. Heatmap generated and saved."
