using CairoMakie
using DelimitedFiles
using LaTeXStrings
using IsingModel

# an example for testing
lattice_sizes = [4, 8, 16, 32]
acorr_maxtaus = Dict(4 => 10, 8 => 50, 16 => 200, 32 => 240)
temperature = 2.269
magnetic_field = 0.0
nsteps_heatbath = 40000
nsteps_eachbin = 1000
nbins = 2100

# Plot the autocorrelation time
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"\tau", ylabel=L"A_{|M|}(τ)", yscale=log10)

# Create a dictionary to store the results
sums = Dict()

for lattice_size in lattice_sizes
    @info """Monte Carlo Simulation of Ferromagnetic Ising Model:
    - temperature = $temperature
    - magnetic_field = $magnetic_field
    - lattice_size = $lattice_size
    """
    model = IsingSpinModel(lattice_size, magnetic_field, 1/temperature)

    # Constructs the initial random spin configuration
    spin = rand([-1,1], model.l, model.l)

    @info """Start the simulation...
    - Monte Carlo steps: ($(nsteps_heatbath) steps heat bath) + ($(nsteps_eachbin) steps each bin) * ($nbins bins))
    """
    result = @time simulate!(model, spin; nsteps_heatbath, nsteps_eachbin, nbins, acorr_maxtau=acorr_maxtaus[lattice_size])

    # Measures the autocorrelation time
    @info "Measures the autocorrelation time..."
    acorr = autocorrelation_time(result)
    filename = joinpath(@__DIR__, "acorr_$lattice_size.dat")
    write(filename, acorr)
    @info "The autocorrelation time is saved to: `$filename`."

    # Calculate the sum of all elements in `acorr`
    total = sum(acorr) - 0.5
    println("The sum of all elements in acorr for lattice size $lattice_size is $total")
    sums[lattice_size] = total
    x_values = 0:(length(acorr)-1) # Create a new x-axis array starting from 0
    lines!(ax, x_values, acorr, label="L=$lattice_size")    # Use the new x-axis array in the plot
    scatter!(ax, x_values, acorr, color=:black) # Add a black dot at each data point
    
end


println(sums)
leg = Legend(fig, ax, "length")
# Add this line to add the legend to the figure
fig[1, 2] = leg
fig

filename = joinpath(@__DIR__, "acor2.png")
save(filename, fig, px_per_unit=2)
@info "The plot is saved to: `$filename`."



