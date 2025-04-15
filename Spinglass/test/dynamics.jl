using Test
using Spinglass: Transverse, iterate_T

@testset "Transverse Model Tests" begin
    # Create a simple test matrix
    J = [0.0 1.0; 1.0 0.0]
    
    # Set up parameters
    n_step = 100
    trials = 1
    beta = exp.(range(log(0.1), log(10), length=n_step))
    
    # Test 1: Gradient descent only (g=1, gama=0)
    @testset "Gradient Descent Configuration" begin
        model_gd = Transverse(
            J, beta, trials, 
            gama=0.0, g=1.0, a_set=2.0, 
            Delta_t=0.3, c0=0.2, 
            seed=42, track_energy=true
        )
        energy_gd, track_gd = iterate_T(model_gd)
        
        @test size(energy_gd) == (trials, n_step)
        @test size(track_gd) == (n_step+1, size(J, 1))
        @test all(track_gd .>= -1.0) && all(track_gd .<= 1.0)
    end
    
    # Test 2: Momentum only (g=0, gama=1)
    @testset "Momentum Configuration" begin
        model_mom = Transverse(
            J, beta, trials, 
            gama=1.0, g=0.0, a_set=2.0, 
            Delta_t=0.3, c0=0.2, 
            seed=42, track_energy=true
        )
        energy_mom, track_mom = iterate_T(model_mom)
        
        @test size(energy_mom) == (trials, n_step)
        @test size(track_mom) == (n_step+1, size(J, 1))
        @test all(track_mom .>= -1.0) && all(track_mom .<= 1.0)
    end
    
    # Test 3: Combined approach (g=0.07, gama=1)
    @testset "Combined Approach" begin
        model_comb = Transverse(
            J, beta, trials, 
            gama=1.0, g=0.07, a_set=2.0, 
            Delta_t=0.3, c0=0.2, 
            seed=42, track_energy=true
        )
        energy_comb, track_comb = iterate_T(model_comb)
        
        @test size(energy_comb) == (trials, n_step)
        @test size(track_comb) == (n_step+1, size(J, 1))
        @test all(track_comb .>= -1.0) && all(track_comb .<= 1.0)
    end
    
    # Test 4: Without energy tracking
    @testset "Without Energy Tracking" begin
        model_no_track = Transverse(
            J, beta, trials, 
            gama=1.0, g=0.07, a_set=2.0, 
            Delta_t=0.3, c0=0.2, 
            seed=42, track_energy=false
        )
        final_energy = iterate_T(model_no_track)
        
        @test size(final_energy) == (1, 1)
        @test typeof(final_energy) <: AbstractArray
    end
    
    # Test 5: Different trial counts
    @testset "Multiple Trials" begin
        multi_trials = 5
        model_multi = Transverse(
            J, beta, multi_trials, 
            gama=1.0, g=0.07, a_set=2.0, 
            Delta_t=0.3, c0=0.2, 
            seed=42, track_energy=true
        )
        energy_multi, _ = iterate_T(model_multi)
        
        @test size(energy_multi) == (multi_trials, n_step)
        @test length(unique(energy_multi[1,:])) > 1  # Energy should change over time
    end
end