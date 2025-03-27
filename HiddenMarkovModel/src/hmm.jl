"""
    HMM{T}

Hidden Markov Model with transition matrix A, emission matrix B, and initial state distribution π.

# Fields
- `A::Matrix{T}`: Transition probability matrix where A[i,j] is the probability of transitioning from state i to state j
- `B::Matrix{T}`: Emission probability matrix where B[i,k] is the probability of emitting observation k from state i
- `π::Vector{T}`: Initial state probability distribution
"""
struct HMM{T<:Real}
    A::Matrix{T}  # Transition matrix
    B::Matrix{T}  # Emission matrix
    π::Vector{T}  # Initial state distribution
    
    function HMM(A::Matrix{T}, B::Matrix{T}, π::Vector{T}) where {T<:Real}
        # Validate dimensions
        n_states = size(A, 1)
        n_observations = size(B, 2)
        
        size(A, 2) == n_states || throw(DimensionMismatch("Transition matrix A must be square"))
        size(B, 1) == n_states || throw(DimensionMismatch("Emission matrix B must have same number of rows as states"))
        length(π) == n_states || throw(DimensionMismatch("Initial distribution π must have same length as number of states"))
        
        # Validate probabilities
        all(isapprox.(sum(A, dims=2), 1)) || @warn "Rows of transition matrix A should sum to 1"
        all(isapprox.(sum(B, dims=2), 1)) || @warn "Rows of emission matrix B should sum to 1"
        isapprox(sum(π), 1) || @warn "Initial distribution π should sum to 1"
        
        new{T}(A, B, π)
    end
end

"""
    forward(hmm::HMM, observations::Vector{Int})

Compute the forward probabilities for a sequence of observations.
Returns the forward probability matrix and the likelihood of the observations.
"""
function forward(hmm::HMM, observations::Vector{Int})
    n_states = length(hmm.π)
    T = length(observations)
    
    # Initialize forward matrix
    α = zeros(n_states, T)
    
    # Initialize first column with initial state * emission probability
    α[:, 1] = hmm.π .* hmm.B[:, observations[1]]
    
    # Forward algorithm using einsum for matrix operations
    for t in 2:T
        # α_t(j) = B_j(o_t) * ∑_i α_{t-1}(i) * A_ij
        α[:, t] = hmm.B[:, observations[t]] .* ein"ij,i->j"(hmm.A, α[:, t-1])
    end
    
    # Likelihood is the sum of the final column
    likelihood = sum(α[:, T])
    
    return α, likelihood
end

"""
    backward(hmm::HMM, observations::Vector{Int})

Compute the backward probabilities for a sequence of observations.
"""
function backward(hmm::HMM, observations::Vector{Int})
    n_states = length(hmm.π)
    T = length(observations)
    
    # Initialize backward matrix
    β = zeros(n_states, T)
    
    # Initialize last column with 1s
    β[:, T] .= 1.0
    
    # Backward algorithm using einsum
    for t in (T-1):-1:1
        # β_t(i) = ∑_j A_ij * B_j(o_{t+1}) * β_{t+1}(j)
        β[:, t] = ein"ij,j,j->i"(hmm.A, hmm.B[:, observations[t+1]], β[:, t+1])
    end
    
    return β
end

"""
    viterbi(hmm::HMM, observations::Vector{Int})

Find the most likely sequence of hidden states given the observations.
"""
function viterbi(hmm::HMM, observations::Vector{Int})
    n_states = length(hmm.π)
    T = length(observations)
    
    # Initialize viterbi matrix (log probabilities)
    V = zeros(n_states, T)
    # Backpointer matrix
    backptr = zeros(Int, n_states, T)
    
    # Initialize first column
    V[:, 1] = log.(hmm.π) .+ log.(hmm.B[:, observations[1]])
    
    # Viterbi algorithm
    for t in 2:T
        for j in 1:n_states
            # Find the most likely previous state
            max_val, max_idx = findmax(V[:, t-1] .+ log.(hmm.A[:, j]))
            V[j, t] = max_val + log(hmm.B[j, observations[t]])
            backptr[j, t] = max_idx
        end
    end
    
    # Backtracking
    path = zeros(Int, T)
    path[T], _ = findmax(V[:, T])
    
    for t in (T-1):-1:1
        path[t] = backptr[path[t+1], t+1]
    end
    
    return path
end

"""
    baum_welch(observations::Vector{Int}, n_states::Int, n_observations::Int; max_iter=100, tol=1e-6)

Learn the parameters of an HMM using the Baum-Welch algorithm (EM for HMMs).
"""
function baum_welch(observations::Vector{Int}, n_states::Int, n_observations::Int; max_iter=100, tol=1e-6)
    # Initialize random HMM parameters
    rng = Random.default_rng()
    
    A = rand(rng, n_states, n_states)
    A = A ./ sum(A, dims=2)  # Normalize rows
    
    B = rand(rng, n_states, n_observations)
    B = B ./ sum(B, dims=2)  # Normalize rows
    
    π = rand(rng, n_states)
    π = π ./ sum(π)  # Normalize
    
    hmm = HMM(A, B, π)
    
    T = length(observations)
    prev_likelihood = -Inf
    
    for iter in 1:max_iter
        # E-step: Compute forward and backward probabilities
        α, likelihood = forward(hmm, observations)
        β = backward(hmm, observations)
        
        # Check convergence
        if abs(likelihood - prev_likelihood) < tol
            break
        end
        prev_likelihood = likelihood
        
        # Compute state probabilities and transition probabilities
        γ = α .* β ./ likelihood  # State probabilities
        
        # Compute ξ (transition probabilities)
        ξ = zeros(n_states, n_states, T-1)
        for t in 1:(T-1)
            # ξ_t(i,j) = P(q_t=i, q_{t+1}=j | O, λ)
            # = α_t(i) * A_ij * B_j(o_{t+1}) * β_{t+1}(j) / P(O|λ)
            denom = likelihood
            for i in 1:n_states
                for j in 1:n_states
                    ξ[i, j, t] = α[i, t] * hmm.A[i, j] * hmm.B[j, observations[t+1]] * β[j, t+1] / denom
                end
            end
        end
        
        # M-step: Update parameters
        # Update initial state distribution
        new_π = γ[:, 1]
        
        # Update transition matrix
        new_A = sum(ξ, dims=3)[:, :, 1] ./ sum(γ[:, 1:end-1], dims=2)
        
        # Update emission matrix
        new_B = zeros(n_states, n_observations)
        for j in 1:n_states
            for k in 1:n_observations
                # Sum γ for all time steps where observation is k
                new_B[j, k] = sum(γ[j, observations .== k]) / sum(γ[j, :])
            end
        end
        
        # Create new HMM with updated parameters
        hmm = HMM(new_A, new_B, new_π)
    end
    
    return hmm
end

"""
    generate_sequence(hmm::HMM, length::Int)

Generate a sequence of observations from the HMM.
"""
function generate_sequence(hmm::HMM, length::Int)
    n_states = size(hmm.A, 1)
    n_observations = size(hmm.B, 2)
    
    # Initialize
    observations = zeros(Int, length)
    states = zeros(Int, length)
    
    # Sample initial state
    states[1] = sample(1:n_states, Weights(hmm.π))
    # Sample initial observation
    observations[1] = sample(1:n_observations, Weights(hmm.B[states[1], :]))
    
    # Generate sequence
    for t in 2:length
        # Sample next state based on transition from current state
        states[t] = sample(1:n_states, Weights(hmm.A[states[t-1], :]))
        # Sample observation from new state
        observations[t] = sample(1:n_observations, Weights(hmm.B[states[t], :]))
    end
    
    return observations, states
end
