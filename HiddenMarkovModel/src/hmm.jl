"""
    HMM{T}

Hidden Markov Model with transition matrix A, emission matrix B, and initial state distribution p0.

# Fields
- `A::Matrix{T}`: Transition probability matrix where A[i,j] is the probability of transitioning from state i to state j
- `B::Matrix{T}`: Emission probability matrix where B[i,k] is the probability of emitting observation k from state i
- `p0::Vector{T}`: Initial state probability distribution
"""
struct HMM{T<:Real}
    A::Matrix{T}  # Transition matrix
    B::Matrix{T}  # Emission matrix
    p0::Vector{T}  # Initial state distribution
    
    function HMM(A::Matrix{T}, B::Matrix{T}, p0::Vector{T}) where {T<:Real}
        # Validate dimensions
        n_states = size(A, 1)
        
        size(A, 2) == n_states || throw(DimensionMismatch("Transition matrix A must be square"))
        size(B, 1) == n_states || throw(DimensionMismatch("Emission matrix B must have same number of rows as states"))
        length(p0) == n_states || throw(DimensionMismatch("Initial distribution p0 must have same length as number of states"))
        
        # Validate probabilities
        all(isapprox.(sum(A, dims=2), 1)) || @warn "Rows of transition matrix A should sum to 1"
        all(isapprox.(sum(B, dims=2), 1)) || @warn "Rows of emission matrix B should sum to 1"
        isapprox(sum(p0), 1) || @warn "Initial distribution p0 should sum to 1"
        
        new{T}(A, B, p0)
    end
end

struct HMMNetwork{T, CT}
    n::Int
    code::CT
    observations::Vector{Int}
    tensors::Vector{Array{T}}
end
p0index(net::HMMNetwork) = 1
aindex(net::HMMNetwork, i::Int) = 1+i
bindex(net::HMMNetwork, i::Int) = net.n+i
function HMMNetwork(hmm::HMM, observations::Vector{Int}; optimizer=GreedyMethod())
    n = length(observations)
    code = EinCode([[1], [[i-1, i] for i in 2:n]..., [[i] for i in 1:n]...], Int[])
    tensors = vcat([hmm.p0], [hmm.A for _ in 1:n-1], [hmm.B[:, o] for o in observations])
    optcode = optimize_code(code, OMEinsum.get_size_dict(code.ixs, (tensors...,)), optimizer)
    return HMMNetwork(n, optcode, observations, tensors)
end
function likelihood_and_gradient(tnet::HMMNetwork)
    return cost_and_gradient(tnet.optcode, (tnet.tensors...,))
end

"""
    forward(hmm::HMM, observations::Vector{Int})

Compute the forward probabilities for a sequence of observations.
Returns the forward probability matrix and the likelihood of the observations.
"""
function forward(hmm::HMM, observations::Vector{Int})
    n_states = length(hmm.p0)
    T = length(observations)
    
    # Initialize forward matrix
    α = zeros(n_states, T)
    
    # Initialize first column with initial state * emission probability
    α[:, 1] = ein"i,i->i"(hmm.p0, hmm.B[:, observations[1]])
    
    # Forward algorithm using einsum for matrix operations
    for t in 2:T
        # α_t(j) = B_j(x_t) * ∑_i α_{t-1}(i) * A_ij
        α[:, t] = ein"j,(ij,i)->j"(hmm.B[:, observations[t]], hmm.A, α[:, t-1])
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
    n_states = length(hmm.p0)
    T = length(observations)
    
    # Initialize backward matrix
    β = zeros(n_states, T)
    
    # Initialize last column with 1s
    β[:, T] .= 1.0
    
    # Backward algorithm using einsum
    for t in (T-1):-1:1
        # β_t(i) = ∑_j A_ij * B_j(x_{t+1}) * β_{t+1}(j)
        β[:, t] = ein"ij,(j,j)->i"(hmm.A, hmm.B[:, observations[t+1]], β[:, t+1])
    end
    
    return β
end

"""
    viterbi(hmm::HMM, observations::Vector{Int})

Find the most likely sequence of hidden states given the observations.
"""
function viterbi(hmm::HMM, observations::Vector{Int})
    n_states = length(hmm.p0)
    T = length(observations)
    
    # Initialize viterbi matrix (log probabilities)
    V = zeros(n_states, T)
    # Backpointer matrix
    backptr = zeros(Int, n_states, T)
    
    # Initialize first column
    V[:, 1] = log.(hmm.p0) .+ log.(hmm.B[:, observations[1]])
    
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
    _, path[T] = findmax(V[:, T])
    
    for t in (T-1):-1:1
        path[t] = backptr[path[t+1], t+1]
    end
    
    return path
end

function state_likelihood(tnet::HMMNetwork, path::Vector{Int})
    prod(vcat(
        [tnet.tensors[p0index(tnet)][path[1]]],
        [tnet.tensors[aindex(tnet, i)][path[i], path[i+1]] for i in 1:length(path)-1],
        [tnet.tensors[bindex(tnet, i)][path[i]] for i in 1:length(path)]
    ))
end

"""
    baum_welch(observations::Vector{Int}, n_states::Int, n_observations::Int; max_iter=100, tol=1e-6)

Learn the parameters of an HMM using the Baum-Welch algorithm (EM for HMMs).
"""
function baum_welch(observations::Vector{Int}, n_states::Int, n_observations::Int; max_iter=100, tol=1e-6)
    # Initialize random HMM parameters
    A = rand(n_states, n_states)
    A = A ./ sum(A, dims=2)  # Normalize rows
    
    B = rand(n_states, n_observations)
    B = B ./ sum(B, dims=2)  # Normalize rows
    
    p0 = rand(n_states)
    p0 = p0 ./ sum(p0)  # Normalize
    
    hmm = HMM(A, B, p0)
    
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
            # ξ_t(i,j) = P(z_t=i, z_{t+1}=j | x, θ)
            # = α_t(i) * A_ij * B_j(x_{t+1}) * β_{t+1}(j) / P(x|θ)
            for i in 1:n_states
                for j in 1:n_states
                    ξ[i, j, t] = α[i, t] * hmm.A[i, j] * hmm.B[j, observations[t+1]] * β[j, t+1] / likelihood
                end
            end
        end

        @show ξ[:,:,1]
        @show gradient(hmm, observations)[2][2]/likelihood .* (hmm.A)
        @show γ[:, 1]
        @show gradient(hmm, observations)[2][1]/likelihood .* (hmm.p0)
        @show γ[:, 2]
        @show gradient(hmm, observations)[2][T+1]/likelihood .* (hmm.B[:, observations[1]])
        
        # M-step: Update parameters
        # Update initial state distribution
        new_p0 = γ[:, 1]
        
        # Update transition matrix
        new_A = dropdims(sum(ξ, dims=3), dims=3) ./ sum(γ[:, 1:end-1], dims=2)
        
        # Update emission matrix
        new_B = zeros(n_states, n_observations)
        for j in 1:n_states
            for k in 1:n_observations
                # Sum γ for all time steps where observation is k
                new_B[j, k] = sum(γ[j, observations .== k]) / sum(γ[j, :])
            end
        end
        
        # Create new HMM with updated parameters
        hmm = HMM(new_A, new_B, new_p0)
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
    states[1] = sample(1:n_states, Weights(hmm.p0))
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

function gradient_descent!(hmm::HMM, observations::Vector{Int}, max_iter::Int=100, 
                         learning_rate::Float64=0.01)
    n = length(observations)
    # Initialize parameters

    # construct the log likelihood function
    code_likelyhood = EinCode([[1], [[i-1, i] for i in 2:n]..., [[i] for i in 1:n]...], Int[])
    tensors = vcat([hmm.p0], [hmm.A for _ in 1:n-1], [hmm.B[:, o] for o in observations])
    optcode = optimize_code(code_likelyhood, OMEinsum.get_size_dict(code_likelyhood.ixs, (tensors...,)), GreedyMethod())

    # compute the log likelihood
    #log_likelihood = code_likelyhood(tensors...)
    for _ in 1:max_iter
        p, gradients = cost_and_gradient(optcode, (tensors...,))
        @info "likelihood: $p"
        gp0, gA, gB = gradients[1] ./ p, sum(gradients[2:end-n]) ./ p, sum(gradients[end-n+1:end]) ./ p
        @info "gradients: $gp0, $gA, $gB"
        # update the parameters
        hmm.A .+= learning_rate .* gA
        hmm.B .+= learning_rate .* gB
        hmm.p0 .+= learning_rate .* gp0
        hmm.A ./= sum(hmm.A; dims=2)
        hmm.B ./= sum(hmm.B; dims=2)
        hmm.p0 ./= sum(hmm.p0)
    end

    return hmm
end

