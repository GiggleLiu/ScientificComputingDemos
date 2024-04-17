using CairoMakie, DelimitedFiles, IsingModel
using Profile


# an example for testing
lattice_sizes = [4, 7, 8, 14, 16, 28, 32, 56]
τ_ranges = Dict(4 => 1:10, 7 => 1:25, 8 => 1:18, 14 => 1:23, 16 => 1:16, 28 => 1:18, 32 => 1:45, 56 => 1:60)
temperature = 2.269
magnetic_field = 0.0
nsteps_heatbath = 1000
nsteps_eachbin = 1000
nbins = 100


# Plot the autocorrelation time
f = Figure()

ax= Axis(f[1,1], xscale = log2, title = string("Scatter plot"), xlabel = "L", ylabel = L"\Theta_{int}",
       )
# Create a dictionary to store the results
sums = Dict()

for lattice_size in lattice_sizes
    model = SwendsenWangModel(lattice_size, magnetic_field, 1/temperature)

    # Constructs the initial random spin configuration
    spin = rand([-1,1], model.l, model.l)

    @info """Start the simulation...
    - Monte Carlo steps: ($(nsteps_heatbath) steps heat bath) + ($(nsteps_eachbin) steps each bin) * ($nbins bins))
    """

    result = @time simulate!(model, spin; nsteps_heatbath=nsteps_heatbath, nbins=nbins, nsteps_eachbin=nsteps_eachbin)
    τ_range = τ_ranges[lattice_size]
    acorr = [autocorrelation_time(result.mstack, t) for t in τ_range]
    filename = joinpath(@__DIR__, "acorr_sw_$lattice_size.dat")
    writedlm(filename, acorr)
    total = sum(acorr) + 0.5
    sums[lattice_size] = total
end


println(sums)

function calculate_Theta_int(L)
    return 0.75 + 0.835 * log(L)
end

# 定义数据点
L_values = [4, 7, 8, 14, 16, 28, 32, 56]
Theta_int_values = calculate_Theta_int.(L_values)
result = sort(collect(sums), by = tuple -> tuple[2])
sorted_values = [tuple[2] for tuple in result]

scatter!(ax, (L_values), sorted_values;
    color = :black,
    markersize = 13,
    marker = :circle,
    linestyle = :none
)

f

lines!(ax, (L_values), Theta_int_values;
color = :red,
)

f

# Save the figure
filename = joinpath(@__DIR__, "acorr_sw.png")
save(filename, f, px_per_unit=2)

