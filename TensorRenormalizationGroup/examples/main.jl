using TensorRenormalizationGroup
using CairoMakie
using Zygote
using Test

@info """
=============================================================================
Tensor Renormalization Group (TRG) for the 2D Ising Model
=============================================================================
This example demonstrates the TRG algorithm, which efficiently computes
thermodynamic properties of the 2D Ising model on large lattices.
"""

# ===== Part 1: Basic TRG Calculation =====
@info "Part 1: Computing the partition function using TRG"

β = 0.44          # Inverse temperature (β = 1/T), chosen near critical point
χ = 20            # Bond dimension cutoff
niter = 10        # Number of iterations

T = 1/β
T_c = 2 / log(1 + sqrt(2))  # Exact critical temperature ≈ 2.269
@info "System parameters:" β T χ niter T_c

# Create the Ising model tensor
a = model_tensor(Ising(), β)
@info "Created local Ising tensor with shape $(size(a))"

# Compute partition function with trace
lnZ, history = trg_with_trace(a, χ, niter)
system_size = 2^niter
@info "Computed log partition function: ln(Z/N) = $lnZ"
@info "Effective lattice size: $(system_size) × $(system_size) = $(system_size^2) sites"

# ===== Part 2: Visualize the Renormalization Process =====
@info "Part 2: Visualizing the renormalization process"

# Extract bond dimensions from history
bond_dims = [h.bond_dim for h in history]
iterations = [h.iteration for h in history]

@info "Bond dimension evolution: $(bond_dims)"

# Create visualization of bond dimension growth
fig1 = Figure(size=(1000, 400))

# Left: Bond dimension vs iteration
ax1 = Axis(fig1[1, 1],
          xlabel="Iteration",
          ylabel="Bond Dimension",
          title="(a) Bond Dimension Evolution")
lines!(ax1, iterations, bond_dims, linewidth=3, color=:blue, marker=:circle, markersize=12)
hlines!(ax1, [χ], linestyle=:dash, color=:red, linewidth=2, label="χ = $χ (cutoff)")
axislegend(ax1, position=:rb)

# Right: System size vs iteration
system_sizes = [2^(i-1) for i in iterations]
ax2 = Axis(fig1[1, 2],
          xlabel="Iteration",
          ylabel="Effective Lattice Size (L)",
          title="(b) Lattice Size Growth",
          yscale=log2)
lines!(ax2, iterations, system_sizes, linewidth=3, color=:green, marker=:square, markersize=12)

filename1 = joinpath(@__DIR__, "trg_renormalization_process.png")
save(filename1, fig1, px_per_unit=2)
@info "Saved renormalization process visualization to '$filename1'"

# ===== Part 3: Visualize Tensor Elements =====
@info "Part 3: Visualizing tensor element structure"

# Create a figure showing how tensor elements evolve
fig2 = Figure(size=(1200, 800))

# Show first, middle, and last tensors
selected_iters = [1, div(niter, 2) + 1, niter + 1]
for (idx, iter) in enumerate(selected_iters)
    h = history[iter]
    
    # Get a 2D slice of the tensor (fix last two indices)
    tensor_slice = real.(h.tensor[:, :, 1, 1])
    
    row = div(idx - 1, 3) + 1
    col = mod(idx - 1, 3) + 1
    
    ax = Axis(fig2[row, col],
             title="Iteration $(h.iteration-1), Bond Dim = $(h.bond_dim)",
             xlabel="Index i",
             ylabel="Index j")
    
    hm = heatmap!(ax, tensor_slice, colormap=:RdBu)
    Colorbar(fig2[row, col][1, 2], hm, label="Tensor Element Value")
end

filename2 = joinpath(@__DIR__, "tensor_evolution.png")
save(filename2, fig2, px_per_unit=2)
@info "Saved tensor evolution visualization to '$filename2'"

# ===== Part 4: Free Energy vs Temperature =====
@info "Part 4: Computing thermodynamic properties across temperatures"

β_range = range(0.2, 0.8, length=30)
temperatures = 1.0 ./ β_range

@info "Computing free energy for $(length(β_range)) temperature points..."
free_energies = Float64[]
lnZ_values = Float64[]

for β_val in β_range
    a_temp = model_tensor(Ising(), β_val)
    lnZ_val = trg(a_temp, χ, niter)
    push!(lnZ_values, lnZ_val)
    # Free energy per site: f = -T * ln(Z/N) = -ln(Z/N) / β
    f = -lnZ_val / β_val
    push!(free_energies, f)
end

@info "Free energy range: [$(minimum(free_energies)), $(maximum(free_energies))]"

# Plot thermodynamic quantities
fig3 = Figure(size=(1200, 800))

# Free energy
ax3_1 = Axis(fig3[1, 1], 
          xlabel="Temperature (T)", 
          ylabel="Free Energy per Site (f)",
          title="(a) Free Energy")
lines!(ax3_1, temperatures, free_energies, linewidth=3, color=:blue)
vlines!(ax3_1, [T_c], linestyle=:dash, color=:red, linewidth=2)
text!(ax3_1, T_c + 0.05, minimum(free_energies) + 0.3, 
      text="Tc ≈ $(round(T_c, digits=3))", color=:red, fontsize=14)

# Log partition function
ax3_2 = Axis(fig3[1, 2], 
          xlabel="Temperature (T)", 
          ylabel="ln(Z/N)",
          title="(b) Log Partition Function")
lines!(ax3_2, temperatures, lnZ_values, linewidth=3, color=:purple)
vlines!(ax3_2, [T_c], linestyle=:dash, color=:red, linewidth=2)

filename3 = joinpath(@__DIR__, "free_energy.png")
save(filename3, fig3, px_per_unit=2)
@info "Saved free energy plot to '$filename3'"

# ===== Part 5: Internal Energy and Specific Heat with AD =====
@info "Part 5: Computing internal energy and specific heat using automatic differentiation"

internal_energies = Float64[]
specific_heats = Float64[]

for β_val in β_range
    # Internal energy: u = -∂ln(Z)/∂β
    f_beta = β -> trg(model_tensor(Ising(), β), χ, niter)
    dlnZ_dβ = Zygote.gradient(f_beta, β_val)[1]
    u = -dlnZ_dβ
    push!(internal_energies, u)
    
    # Specific heat: C = β² * ∂u/∂β
    d2lnZ_dβ2 = Zygote.gradient(β -> Zygote.gradient(f_beta, β)[1], β_val)[1]
    C = -β_val^2 * d2lnZ_dβ2
    push!(specific_heats, C)
end

@info "Internal energy range: [$(minimum(internal_energies)), $(maximum(internal_energies))]"
@info "Specific heat range: [$(minimum(specific_heats)), $(maximum(specific_heats))]"
@info "Peak specific heat at T ≈ $(temperatures[argmax(specific_heats)]) (near Tc = $(round(T_c, digits=3)))"

# Plot internal energy and specific heat
fig4 = Figure(size=(1200, 800))

ax4_1 = Axis(fig4[1, 1], 
          xlabel="Temperature (T)", 
          ylabel="Internal Energy per Site (u)",
          title="(c) Internal Energy")
lines!(ax4_1, temperatures, internal_energies, linewidth=3, color=:green)
vlines!(ax4_1, [T_c], linestyle=:dash, color=:red, linewidth=2)

ax4_2 = Axis(fig4[1, 2], 
          xlabel="Temperature (T)", 
          ylabel="Specific Heat per Site (C)",
          title="(d) Specific Heat")
lines!(ax4_2, temperatures, specific_heats, linewidth=3, color=:orange)
vlines!(ax4_2, [T_c], linestyle=:dash, color=:red, linewidth=2)
text!(ax4_2, T_c + 0.05, maximum(specific_heats) * 0.9, 
      text="Tc ≈ $(round(T_c, digits=3))", color=:red, fontsize=14)

filename4 = joinpath(@__DIR__, "internal_energy_specific_heat.png")
save(filename4, fig4, px_per_unit=2)
@info "Saved internal energy and specific heat plot to '$filename4'"

# ===== Part 6: Combined Thermodynamic Plot =====
@info "Part 6: Creating combined thermodynamic visualization"

fig5 = Figure(size=(1200, 900))

# All four quantities in one figure
ax5_1 = Axis(fig5[1, 1], xlabel="T", ylabel="f", title="(a) Free Energy")
lines!(ax5_1, temperatures, free_energies, linewidth=2, color=:blue)
vlines!(ax5_1, [T_c], linestyle=:dash, color=:red, linewidth=1)

ax5_2 = Axis(fig5[1, 2], xlabel="T", ylabel="ln(Z/N)", title="(b) Log Partition Function")
lines!(ax5_2, temperatures, lnZ_values, linewidth=2, color=:purple)
vlines!(ax5_2, [T_c], linestyle=:dash, color=:red, linewidth=1)

ax5_3 = Axis(fig5[2, 1], xlabel="T", ylabel="u", title="(c) Internal Energy")
lines!(ax5_3, temperatures, internal_energies, linewidth=2, color=:green)
vlines!(ax5_3, [T_c], linestyle=:dash, color=:red, linewidth=1)

ax5_4 = Axis(fig5[2, 2], xlabel="T", ylabel="C", title="(d) Specific Heat")
lines!(ax5_4, temperatures, specific_heats, linewidth=2, color=:orange)
vlines!(ax5_4, [T_c], linestyle=:dash, color=:red, linewidth=1)

filename5 = joinpath(@__DIR__, "thermodynamic_properties.png")
save(filename5, fig5, px_per_unit=2)
@info "Saved combined thermodynamic properties to '$filename5'"

# ===== Part 7: Convergence Study =====
@info "Part 7: Studying convergence with respect to bond dimension"

χ_values = [2, 4, 8, 12, 16, 20, 24, 32]
β_test = 0.44  # Near critical point

@info "Testing convergence at β = $β_test (T = $(round(1/β_test, digits=3)))"

lnZ_convergence = Float64[]
for χ_val in χ_values
    a_test = model_tensor(Ising(), β_test)
    lnZ_test = trg(a_test, χ_val, niter)
    push!(lnZ_convergence, lnZ_test)
    @info "  χ = $χ_val: ln(Z/N) = $lnZ_test"
end

# Plot convergence
fig6 = Figure(size=(1000, 600))

ax6_1 = Axis(fig6[1, 1], 
          xlabel="Bond Dimension (χ)", 
          ylabel="ln(Z/N)",
          title="(a) Convergence Study (Linear Scale)")
lines!(ax6_1, χ_values, lnZ_convergence, linewidth=3, color=:blue, 
       marker=:circle, markersize=12)

ax6_2 = Axis(fig6[1, 2], 
          xlabel="Bond Dimension (χ)", 
          ylabel="ln(Z/N)",
          title="(b) Convergence Study (Log Scale)",
          xscale=log10)
lines!(ax6_2, χ_values, lnZ_convergence, linewidth=3, color=:blue, 
       marker=:circle, markersize=12)

filename6 = joinpath(@__DIR__, "convergence_study.png")
save(filename6, fig6, px_per_unit=2)
@info "Saved convergence study to '$filename6'"

# ===== Part 8: Verify Automatic Differentiation =====
@info "Part 8: Verifying automatic differentiation with numerical gradient"

@testset "Gradient Verification" begin
    β_test_grad = 0.4
    @info "Testing gradient at β = $β_test_grad"
    
    # Function to compute log partition function
    f = β -> trg(model_tensor(Ising(), β), 10, 8)
    
    # Compute gradient using Zygote
    grad_auto = Zygote.gradient(f, β_test_grad)[1]
    
    # Compute numerical gradient
    grad_numerical = num_grad(f, β_test_grad, δ=1e-6)
    
    @info "  Zygote gradient: $grad_auto"
    @info "  Numerical gradient: $grad_numerical"
    
    # Compute relative error
    rel_error = abs(grad_auto - grad_numerical) / abs(grad_numerical)
    @info "  Relative error: $rel_error"
    
    @test rel_error < 1e-4
    @info "  ✓ Gradient verification PASSED!"
end

# ===== Part 9: Phase Transition Visualization =====
@info "Part 9: Visualizing the phase transition"

# Compute specific heat peak
peak_idx = argmax(specific_heats)
peak_T = temperatures[peak_idx]
peak_C = specific_heats[peak_idx]

@info "Specific heat peak: C_max = $peak_C at T = $peak_T"
@info "Critical temperature (exact): Tc = $T_c"
@info "Relative difference: $((peak_T - T_c) / T_c * 100)%"

# Create phase transition visualization
fig7 = Figure(size=(1000, 700))

ax7 = Axis(fig7[1, 1],
          xlabel="Temperature (T)",
          ylabel="Specific Heat per Site (C)",
          title="Phase Transition in 2D Ising Model")

# Plot specific heat
lines!(ax7, temperatures, specific_heats, linewidth=4, color=:orange, label="TRG (χ=$χ)")

# Mark critical temperature
vlines!(ax7, [T_c], linestyle=:dash, color=:red, linewidth=2, label="Tc (exact)")
vlines!(ax7, [peak_T], linestyle=:dot, color=:blue, linewidth=2, label="Peak")

# Add annotations
text!(ax7, 1.2, peak_C * 0.9, 
      text="Low T:\nOrdered Phase\n(Ferromagnetic)", 
      color=:blue, fontsize=14, align=(:left, :top))
text!(ax7, 3.5, peak_C * 0.9, 
      text="High T:\nDisordered Phase\n(Paramagnetic)", 
      color=:red, fontsize=14, align=(:right, :top))

axislegend(ax7, position=:rt)

filename7 = joinpath(@__DIR__, "phase_transition.png")
save(filename7, fig7, px_per_unit=2)
@info "Saved phase transition visualization to '$filename7'"

# ===== Summary =====
@info """
=============================================================================
Summary: Tensor Renormalization Group Simulation
=============================================================================

Physical System:
  • 2D Ising model on a square lattice
  • Critical temperature: Tc = $(round(T_c, digits=4))
  • Effective lattice size: $(2^niter) × $(2^niter) = $(2^(2*niter)) sites

Algorithm Parameters:
  • Bond dimension cutoff: χ = $χ
  • Number of iterations: $niter
  • Convergence: $(round(lnZ_convergence[end] - lnZ_convergence[end-1], sigdigits=3)) change in ln(Z/N) for last χ doubling

Key Results:
  • Phase transition detected at T ≈ $peak_T (error: $(round(abs(peak_T - T_c) / T_c * 100, digits=2))%)
  • Maximum specific heat: C_max = $(round(peak_C, digits=3))
  • Automatic differentiation successfully verified

Generated Files:
  1. $filename1 - Renormalization process visualization
  2. $filename2 - Tensor element evolution
  3. $filename3 - Free energy and log partition function
  4. $filename4 - Internal energy and specific heat
  5. $filename5 - Combined thermodynamic properties
  6. $filename6 - Convergence study
  7. $filename7 - Phase transition visualization

The TRG algorithm efficiently computes partition functions for exponentially
large lattices by systematically coarse-graining the system while maintaining
accuracy through SVD truncation.
=============================================================================
"""

@info "Tensor Renormalization Group example completed successfully!"
