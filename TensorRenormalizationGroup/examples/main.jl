using TensorRenormalizationGroup
using CairoMakie
using Zygote

println("\n" * "="^70)
println("Tensor Renormalization Group (TRG) for the 2D Ising Model")
println("="^70)

# ===== Basic TRG Calculation =====
println("\n=== Basic TRG Calculation ===")

# Parameters
β = 0.4          # Inverse temperature (β = 1/T)
χ = 5            # Bond dimension cutoff
niter = 5        # Number of iterations

println("Parameters:")
println("  Inverse temperature (β) = $β")
println("  Bond dimension (χ) = $χ")
println("  Number of iterations = $niter")

# Create the Ising model tensor
a = model_tensor(Ising(), β)
println("\nLocal tensor shape: $(size(a))")
println("Each index corresponds to a spin: {1,2} → {-1,+1}")

# Compute the logarithm of the partition function
lnZ = trg(a, χ, niter)
println("\nLog partition function per site: ln(Z/N) = $lnZ")
println("Partition function per site: Z/N = $(exp(lnZ))")

# The system size after niter iterations
N = 2^niter
println("System size: $N × $N = $(N^2) sites")

# ===== Free Energy vs Temperature =====
println("\n=== Free Energy as a Function of Temperature ===")

# Define temperature range
β_range = range(0.1, 1.0, length=20)
temperatures = 1.0 ./ β_range

println("Computing free energy for $(length(β_range)) temperature points...")
free_energies = Float64[]

for β_val in β_range
    a = model_tensor(Ising(), β_val)
    lnZ_val = trg(a, χ, niter)
    # Free energy per site: f = -T * ln(Z/N) = -ln(Z/N) / β
    f = -lnZ_val / β_val
    push!(free_energies, f)
end

println("Free energy range: [$(minimum(free_energies)), $(maximum(free_energies))]")

# Plot free energy vs temperature
fig1 = Figure(size=(800, 600))
ax1 = Axis(fig1[1, 1], 
          xlabel="Temperature (T)", 
          ylabel="Free Energy per Site (f)",
          title="Free Energy of 2D Ising Model (TRG)")
lines!(ax1, temperatures, free_energies, linewidth=3, color=:blue)
scatter!(ax1, temperatures, free_energies, markersize=10, color=:blue)

# Add vertical line at critical temperature
T_c = 2 / log(1 + sqrt(2))  # Exact critical temperature
vlines!(ax1, [T_c], linestyle=:dash, color=:red, linewidth=2)
text!(ax1, T_c + 0.05, minimum(free_energies) + 0.5, 
      text="Tc ≈ $(round(T_c, digits=3))", color=:red, fontsize=14)

filename1 = joinpath(@__DIR__, "free_energy_vs_temperature.png")
save(filename1, fig1, px_per_unit=2)
println("\nSaved free energy plot to '$filename1'")

# ===== Internal Energy with Automatic Differentiation =====
println("\n=== Internal Energy via Automatic Differentiation ===")

# Internal energy per site: u = ∂(β*f)/∂β = -∂ln(Z)/∂β
println("Computing internal energy using Zygote for automatic differentiation...")

internal_energies = Float64[]
for β_val in β_range
    # Define function to compute log partition function
    f_beta = β -> trg(model_tensor(Ising(), β), χ, niter)
    
    # Compute derivative using Zygote
    dlnZ_dβ = Zygote.gradient(f_beta, β_val)[1]
    
    # Internal energy per site
    u = -dlnZ_dβ
    push!(internal_energies, u)
end

println("Internal energy range: [$(minimum(internal_energies)), $(maximum(internal_energies))]")

# Plot internal energy vs temperature
fig2 = Figure(size=(800, 600))
ax2 = Axis(fig2[1, 1], 
          xlabel="Temperature (T)", 
          ylabel="Internal Energy per Site (u)",
          title="Internal Energy of 2D Ising Model (TRG + AD)")
lines!(ax2, temperatures, internal_energies, linewidth=3, color=:green)
scatter!(ax2, temperatures, internal_energies, markersize=10, color=:green)

# Add vertical line at critical temperature
vlines!(ax2, [T_c], linestyle=:dash, color=:red, linewidth=2)
text!(ax2, T_c + 0.05, minimum(internal_energies) + 0.2, 
      text="Tc ≈ $(round(T_c, digits=3))", color=:red, fontsize=14)

filename2 = joinpath(@__DIR__, "internal_energy_vs_temperature.png")
save(filename2, fig2, px_per_unit=2)
println("\nSaved internal energy plot to '$filename2'")

# ===== Specific Heat =====
println("\n=== Specific Heat via Automatic Differentiation ===")

println("Computing specific heat (second derivative)...")

specific_heats = Float64[]
for β_val in β_range
    # Specific heat: C = β² * ∂²ln(Z)/∂β²
    f_beta = β -> trg(model_tensor(Ising(), β), χ, niter)
    
    # Compute second derivative
    # C = -β² * d²ln(Z)/dβ² = β² * du/dβ
    d2lnZ_dβ2 = Zygote.gradient(β -> Zygote.gradient(f_beta, β)[1], β_val)[1]
    C = -β_val^2 * d2lnZ_dβ2
    
    push!(specific_heats, C)
end

println("Specific heat range: [$(minimum(specific_heats)), $(maximum(specific_heats))]")

# Plot specific heat vs temperature
fig3 = Figure(size=(800, 600))
ax3 = Axis(fig3[1, 1], 
          xlabel="Temperature (T)", 
          ylabel="Specific Heat per Site (C)",
          title="Specific Heat of 2D Ising Model (TRG + AD)")
lines!(ax3, temperatures, specific_heats, linewidth=3, color=:purple)
scatter!(ax3, temperatures, specific_heats, markersize=10, color=:purple)

# Add vertical line at critical temperature
vlines!(ax3, [T_c], linestyle=:dash, color=:red, linewidth=2)
text!(ax3, T_c + 0.05, maximum(specific_heats) * 0.5, 
      text="Tc ≈ $(round(T_c, digits=3))", color=:red, fontsize=14)

filename3 = joinpath(@__DIR__, "specific_heat_vs_temperature.png")
save(filename3, fig3, px_per_unit=2)
println("\nSaved specific heat plot to '$filename3'")

# ===== Combined Plot =====
println("\n=== Creating Combined Plot ===")

fig4 = Figure(size=(1200, 400))

# Free energy subplot
ax4_1 = Axis(fig4[1, 1], 
            xlabel="Temperature (T)", 
            ylabel="Free Energy (f)",
            title="(a) Free Energy")
lines!(ax4_1, temperatures, free_energies, linewidth=2, color=:blue)
vlines!(ax4_1, [T_c], linestyle=:dash, color=:red, linewidth=1)

# Internal energy subplot
ax4_2 = Axis(fig4[1, 2], 
            xlabel="Temperature (T)", 
            ylabel="Internal Energy (u)",
            title="(b) Internal Energy")
lines!(ax4_2, temperatures, internal_energies, linewidth=2, color=:green)
vlines!(ax4_2, [T_c], linestyle=:dash, color=:red, linewidth=1)

# Specific heat subplot
ax4_3 = Axis(fig4[1, 3], 
            xlabel="Temperature (T)", 
            ylabel="Specific Heat (C)",
            title="(c) Specific Heat")
lines!(ax4_3, temperatures, specific_heats, linewidth=2, color=:purple)
vlines!(ax4_3, [T_c], linestyle=:dash, color=:red, linewidth=1)

filename4 = joinpath(@__DIR__, "thermodynamic_quantities.png")
save(filename4, fig4, px_per_unit=2)
println("\nSaved combined thermodynamic plot to '$filename4'")

# ===== Convergence Study =====
println("\n=== Convergence Study ===")

println("Testing convergence with different bond dimensions...")

χ_values = [2, 4, 8, 16, 32]
β_test = 0.44  # Near critical point

println("Testing at β = $β_test (T = $(round(1/β_test, digits=3)))")

lnZ_values = Float64[]
for χ_val in χ_values
    a = model_tensor(Ising(), β_test)
    lnZ_val = trg(a, χ_val, niter)
    push!(lnZ_values, lnZ_val)
    println("  χ = $χ_val: ln(Z/N) = $lnZ_val")
end

# Plot convergence
fig5 = Figure(size=(800, 600))
ax5 = Axis(fig5[1, 1], 
          xlabel="Bond Dimension (χ)", 
          ylabel="Log Partition Function per Site",
          title="Convergence of TRG (β = $β_test)",
          xscale=log10)
lines!(ax5, χ_values, lnZ_values, linewidth=3, color=:orange, marker=:circle, markersize=15)

filename5 = joinpath(@__DIR__, "convergence_study.png")
save(filename5, fig5, px_per_unit=2)
println("\nSaved convergence plot to '$filename5'")

# ===== Numerical Gradient Check =====
println("\n=== Numerical Gradient Verification ===")

β_test2 = 0.4
println("Verifying automatic differentiation at β = $β_test2")

# Function to compute log partition function
f = β -> trg(model_tensor(Ising(), β), 5, 5)

# Compute gradient using Zygote
grad_auto = Zygote.gradient(f, β_test2)[1]
println("  Automatic gradient (Zygote): $(grad_auto)")

# Compute numerical gradient
grad_numerical = num_grad(f, β_test2, δ=1e-6)
println("  Numerical gradient: $(grad_numerical)")

# Compute relative error
rel_error = abs(grad_auto - grad_numerical) / abs(grad_numerical)
println("  Relative error: $(rel_error)")

if rel_error < 1e-4
    println("  ✓ Gradient verification PASSED!")
else
    println("  ✗ Gradient verification FAILED!")
end

# ===== Summary =====
println("\n" * "="^70)
println("Summary")
println("="^70)
println("""
The Tensor Renormalization Group (TRG) method efficiently computes
thermodynamic properties of the 2D Ising model on large lattices.

Key Results:
  • Critical Temperature: Tc ≈ $(round(T_c, digits=4))
  • TRG systematically approximates the partition function
  • Automatic differentiation enables computation of derivatives
  • Specific heat shows peak near critical temperature

Generated Plots:
  1. $filename1
  2. $filename2
  3. $filename3
  4. $filename4
  5. $filename5

The TRG algorithm:
  • Coarse-grains the lattice by a factor of 2 per iteration
  • Maintains approximate accuracy via SVD truncation
  • Scales efficiently to large systems (exponentially large lattices)
  • Works for both real and complex tensors
""")

println("\n" * "="^70)
println("Example completed successfully!")
println("="^70 * "\n")

