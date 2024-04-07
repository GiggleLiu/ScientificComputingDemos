using CairoMakie
using DelimitedFiles
using LaTeXStrings
using IsingModel

# an example for testing
lattice_sizes = [4, 8, 16, 32]
temperatures = [2.269]
# Create a dictionary of acorr_maxtaus for each temperature
acorr_maxtaus_dict = Dict(
    2.269 => Dict(4 => 3,  8 => 3, 16 => 4, 32 => 4)
)
magnetic_field = 0.0

nsteps_heatbath = 10000
nsteps_eachbin = 100
nbins = 1000

 
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"L", ylabel=L"A_{int|M|}(τ)", xscale=log2)

for temperature in temperatures 
    # Get the acorr_maxtaus for the current temperature
    acorr_maxtaus = acorr_maxtaus_dict[temperature]
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
    filename = joinpath(@__DIR__, "plot_acorr_$lattice_size.dat")
    write(filename, acorr)
    @info "The autocorrelation time is saved to: `$filename`."

    # Calculate the sum of all elements in `acorr`
    total = sum(acorr) - 0.5
        println("The sum of all elements in acorr for lattice size $lattice_size is $total")
        sums[lattice_size] = total
    end

    sorted_keys = sort(collect(keys(sums)))
    A = [sums[key] for key in sorted_keys]

    # Add scatter plot for current temperature
    scatter!(ax, lattice_sizes, A;
        color = temperature == 2.269 ? :red : :blue,  # Use different colors for different temperatures
        markersize = 10,
        marker = :circle,
        label = "T=$temperature"  # Add temperature to the label
    )

    # Add lines between the points
    lines!(ax, lattice_sizes, A;
        color = temperature == 2.269 ? :red : :blue  # Use different colors for different temperatures
    )
end

# Calculate Theta_int
function calculate_Theta_int(L)
    return 0.75 + 0.85 * log(L)
end

# define data points
Theta_int_values = calculate_Theta_int.(lattice_sizes)

# Add lines for Theta_int_values
lines!(ax, lattice_sizes, Theta_int_values;
    color = :green,
)

fig

# Save the figure
save(joinpath(@__DIR__, "acorr4.png"), fig)


