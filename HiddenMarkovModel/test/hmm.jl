using HiddenMarkovModel, Test

@testset "HMM Basics" begin
    # Create a simple HMM with 2 states and 3 possible observations
    A = [0.7 0.3; 0.4 0.6]  # Transition matrix
    B = [0.1 0.4 0.5; 0.7 0.2 0.1]  # Emission matrix
    p0 = [0.6, 0.4]  # Initial state distribution
    
    hmm = HMM(A, B, p0)
    
    @test size(hmm.A) == (2, 2)
    @test size(hmm.B) == (2, 3)
    @test length(hmm.p0) == 2
    @test isapprox(sum(hmm.A, dims=2), ones(2, 1))
    @test isapprox(sum(hmm.B, dims=2), ones(2, 1))
    @test isapprox(sum(hmm.p0), 1.0)
end

@testset "Forward Algorithm" begin
    A = [0.7 0.3; 0.4 0.6]
    B = [0.1 0.4 0.5; 0.7 0.2 0.1]
    p0 = [0.6, 0.4]
    hmm = HMM(A, B, p0)
    
    # Test with a simple observation sequence
    observations = [1, 3, 2]
    α, likelihood = forward(hmm, observations)
    
    @test size(α) == (2, 3)
    @test likelihood > 0
    @test likelihood <= 1.0
    
    # First column should be p0 .* B[:, observations[1]]
    @test isapprox(α[:, 1], p0 .* B[:, observations[1]])
end

@testset "Backward Algorithm" begin
    A = [0.7 0.3; 0.4 0.6]
    B = [0.1 0.4 0.5; 0.7 0.2 0.1]
    p0 = [0.6, 0.4]
    hmm = HMM(A, B, p0)
    
    observations = [1, 3, 2]
    β = backward(hmm, observations)
    
    @test size(β) == (2, 3)
    # Last column should be all ones
    @test all(isapprox.(β[:, end], 1.0))
end

@testset "Viterbi Algorithm" begin
    A = [0.7 0.3; 0.4 0.6]
    B = [0.1 0.4 0.5; 0.7 0.2 0.1]
    p0 = [0.6, 0.4]
    hmm = HMM(A, B, p0)
    
    observations = [1, 3, 2]
    best_path = viterbi(hmm, observations)
    
    @test length(best_path) == length(observations)
    net = HMMNetwork(hmm, observations)
    @test best_path ≈ [2, 1, 1]
    for _ in 1:10
        path = rand(1:2, length(observations))
        @test HiddenMarkovModel.state_likelihood(net, best_path) >= HiddenMarkovModel.state_likelihood(net, path)
    end
    @test all(1 .<= best_path .<= 2)  # States should be 1 or 2
end

@testset "Sequence Generation" begin
    A = [0.7 0.3; 0.4 0.6]
    B = [0.0 0.0 1.0; 0.9 0.0 0.1]
    p0 = [0.6, 0.4]
    hmm = HMM(A, B, p0)
    
    seq_length = 100
    observations, states = generate_sequence(hmm, seq_length)
    @test forward(hmm, rand(1:3, 100))[2] < forward(hmm, observations)[2]
    @test forward(hmm, observations)[2] > 0.0
    
    @test length(observations) == seq_length
    @test length(states) == seq_length
    @test all(1 .<= observations .<= 3)  # Observations should be 1, 2, or 3
    @test all(1 .<= states .<= 2)  # States should be 1 or 2
end

@testset "Baum-Welch Algorithm" begin
    # Create a simple HMM
    A = [0.6 0.4; 0.3 0.7]
    B = [0.0 0.0 1.0; 0.9 0.0 0.1]
    p0 = [0.5, 0.5]
    true_hmm = HMM(A, B, p0)
    
    # Generate a sequence from the true HMM
    seq_length = 200
    observations, _ = generate_sequence(true_hmm, seq_length)
    
    # Create an initial guess HMM
    A_init = [0.5 0.5; 0.5 0.5]
    B_init = [1/3 1/3 1/3; 1/3 1/3 1/3]
    p0_init = [0.5, 0.5]
    initial_hmm = HMM(A_init, B_init, p0_init)
    
    # Train the model
    trained_hmm = baum_welch(observations, 2, 3, max_iter=5)
    p_true = forward(true_hmm, observations)[2]
    p_trained = forward(trained_hmm, observations)[2]
    p_initial = forward(initial_hmm, observations)[2]
    @show p_true, p_trained, p_initial
    @test p_true > p_trained > p_initial
    
    @test size(trained_hmm.A) == size(true_hmm.A)
    @test size(trained_hmm.B) == size(true_hmm.B)
    @test length(trained_hmm.p0) == length(true_hmm.p0)
    
    # Check that probabilities are valid
    @test all(0 .<= trained_hmm.A .<= 1)
    @test all(0 .<= trained_hmm.B .<= 1)
    @test all(0 .<= trained_hmm.p0 .<= 1)
    @test isapprox(sum(trained_hmm.p0), 1.0)
    @test all(isapprox.(sum(trained_hmm.A, dims=2), ones(2, 1)))
    @test all(isapprox.(sum(trained_hmm.B, dims=2), ones(2, 1)))
end

@testset "Gradient Descent" begin
    # Create a simple HMM
    A = [0.6 0.4; 0.3 0.7]
    B = [0.0 0.0 1.0; 0.9 0.0 0.1]
    p0 = [0.5, 0.5]
    true_hmm = HMM(A, B, p0)

    # Generate a sequence from the true HMM
    seq_length = 200
    observations, _ = generate_sequence(true_hmm, seq_length)
    
    # Create an initial guess HMM
    A_init = [0.5 0.5; 0.5 0.5]
    B_init = [1/3 1/3 1/3; 1/3 1/3 1/3]
    p0_init = [0.5, 0.5]
    initial_hmm = HMM(A_init, B_init, p0_init)

    forward_likelihood = forward(initial_hmm, observations)
    trained_hmm = HiddenMarkovModel.gradient_descent!(deepcopy(initial_hmm), observations)
    
    p_true = forward(true_hmm, observations)[2]
    p_initial = forward(initial_hmm, observations)[2]
    p_trained = forward(trained_hmm, observations)[2]

    @show p_true, p_initial, p_trained
    @test p_true > p_trained > p_initial
end
