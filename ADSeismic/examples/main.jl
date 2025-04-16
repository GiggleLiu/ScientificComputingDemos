using ADSeismic, CairoMakie
using ADSeismic: SeismicState, treeverse_step, TreeverseLog
using LinearAlgebra

nx = ny = 50
N = 1000
c = 1000*ones(nx+2, ny+2)

# gradient
param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
    Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=N)
src = size(c) .÷ 2 .- 1
srcv = Ricker(param, 100.0, 500.0)

# Plot the Ricker wavelet (source time function)
function plot_waveform(srcv)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time (s)", ylabel="Amplitude", title="Ricker Wavelet")
    lines!(ax, srcv, linewidth=2, color=:blue)
    save(joinpath(@__DIR__, "ricker_wavelet.png"), fig)
    fig
end
 
plot_waveform(srcv)

s1 = SeismicState([randn(nx+2,ny+2) for i=1:4]..., Ref(2))
s0 = SeismicState(Float64, nx, ny)

# setup initial gradient
gn = SeismicState(Float64, nx, ny)
gn.u[45,45] = 1.0

log = TreeverseLog()
res0 = solve(param, src, srcv, copy(c))
res, (gx, gsrcv, gc) = treeverse_gradient(s0,
            x -> (gn, zero(srcv), zero(c));  # return the last gradient
            param=param, c=copy(c), src,
            srcv=srcv, δ=50, logger=log);

# Visualize the gradient with respect to the initial state (g_tv_x)
function visualize_gradient(g)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
        xlabel="X (grid points)", 
        ylabel="Y (grid points)", 
        title="Gradient with respect to initial state")
    
    # Create heatmap of the gradient
    hm = heatmap!(ax, g, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="Gradient magnitude")
    
    save(joinpath(@__DIR__, "gradient_visualization.png"), fig)
    return fig
end

# Display the gradient visualization
visualize_gradient(gc)

# Import Optimizers.jl for Adam optimizer
using Optimisers

# Define the loss function that we want to minimize
function loss_function(c_model)
    # Solve the forward problem with the current velocity model
    result = ADSeismic.solve_final(param, src, srcv, c_model)
    return result[45, 45]
end

# Function to compute gradient using treeverse
function compute_gradient(c_model)
    gn = SeismicState(Float64, nx, ny)
    gn.u[45,45] = 1.0
    # Solve and get gradient
    res, (_, _, gradient) = treeverse_gradient(s0,
                x -> (gn, zero(srcv), zero(c_model));
                param=param, c=c_model, src=src,
                srcv=srcv, δ=50, logger=log)
    @show res.u[45, 45]
    return gradient
end

# Setup Adam optimizer
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
adam = Adam(learning_rate)

# Initial velocity model
c_current = copy(c)

# Number of optimization iterations
num_iterations = 50

# Store loss history for plotting
loss_history = Float64[]

# Function to perform optimization using Adam optimizer
function optimize_model(c_initial, num_iterations, learning_rate=0.01)
    # Setup Adam optimizer
    opt_state = Optimisers.setup(Adam(learning_rate), c_initial)
    c_current = copy(c_initial)
    loss_values = Float64[]
    
    for iter in 1:num_iterations
        # Compute current loss
        current_loss = loss_function(c_current)
        push!(loss_values, current_loss)
        
        println("Iteration $iter, Loss: $current_loss")
        
        # Compute gradient
        gradient = compute_gradient(c_current)
        
        # Update velocity model using Adam optimizer
        @show norm(gradient)
        Optimisers.update!(opt_state, c_current, gradient)
    end
    
    return c_current, loss_values
end

# Run optimization
c_current, loss_history = optimize_model(c, num_iterations)
# Plot loss history
function plot_loss_history(loss_history)
    fig = Figure(size=(800, 400))
    ax = Axis(fig[1, 1], 
        xlabel="Iteration", 
        ylabel="Loss", 
        title="Optimization Loss History")
    
    lines!(ax, 1:length(loss_history), loss_history, linewidth=2, color=:blue)
    
    save(joinpath(@__DIR__, "optimization_loss_history.png"), fig)
    return fig
end

# Display the loss history
plot_loss_history(loss_history)

# Visualize final optimized model
function visualize_final_model(initial_model, optimized_model)
    fig = Figure(size=(1200, 600))
    
    ax1 = Axis(fig[1, 1], 
        xlabel="X (grid points)", 
        ylabel="Y (grid points)", 
        title="Initial Velocity Model")
    
    ax2 = Axis(fig[1, 2], 
        xlabel="X (grid points)", 
        ylabel="Y (grid points)", 
        title="Optimized Velocity Model")
    
    hm1 = heatmap!(ax1, initial_model, colormap=:viridis)
    hm2 = heatmap!(ax2, optimized_model, colormap=:viridis)
    
    Colorbar(fig[1, 3], hm2, label="Velocity (m²/s²)")
    
    save(joinpath(@__DIR__, "final_comparison.png"), fig)
    return fig
end

# Compare initial and final models
visualize_final_model(c2, c_current)

