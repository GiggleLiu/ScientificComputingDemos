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
    return sum(abs2, result[45, 45])
end

# Function to compute gradient using treeverse
function compute_gradient(c_model)
    function gn(x)
        gx = SeismicState(Float64, nx, ny)
        gx.u[40:end,40:end] .= 2 .* x.u[40:end, 40:end]
        return (gx, zero(srcv), zero(c_model))
    end
    # Solve and get gradient
    res, (_, _, gradient) = treeverse_gradient(s0, gn;
                param=param, c=c_model, src=src,
                srcv=srcv, δ=50, logger=log)
    @show norm(res.u[40:end, 40:end])
    return gradient
end

# Create animation of wave propagation
function animate_wavefield(wavefield)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], 
        xlabel="X (grid points)", 
        ylabel="Y (grid points)", 
        title="Acoustic Wave Propagation")
    
    # Get min/max values for consistent colormap scaling
    vmin, vmax = extrema(wavefield)
    
    # Create initial heatmap
    hm = heatmap!(ax, wavefield[:, :, 1]', colormap=:seismic, 
                  colorrange=(vmin, vmax))
    Colorbar(fig[1, 2], hm, label="Amplitude")
    
    # Create animation
    framerate = 30
    step = max(1, div(size(wavefield, 3), 300))  # Limit to ~300 frames for performance
    
    record(fig, joinpath(@__DIR__, "wave_propagation.mp4"), 1:step:size(wavefield, 3); framerate=framerate) do i
        hm[3][] = wavefield[:, :, i]'
        ax.title = "Acoustic Wave Propagation (t = $(i))"
    end
    
    @info "Animation saved to: ", joinpath(@__DIR__, "wave_propagation.mp4")
end

# Initial velocity model
c_current = copy(c)

# Store loss history for plotting
loss_history = Float64[]

# Function to perform optimization using Adam optimizer
function optimize_model(c_initial, num_iterations, learning_rate=10.0)
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
c_current, loss_history = optimize_model(c, 500)
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

res = solve(param, src, srcv, c_current)
# Generate the animation
animate_wavefield(res)

