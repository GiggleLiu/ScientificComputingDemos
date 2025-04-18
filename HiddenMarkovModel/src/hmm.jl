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
arange(net::HMMNetwork) = 2:net.n
brange(net::HMMNetwork) = net.n+1:2*net.n
function HMMNetwork(hmm::HMM, observations::Vector{Int}; optimizer=GreedyMethod())
    n = length(observations)
    code = EinCode([[1], [[i-1, i] for i in 2:n]..., [[i] for i in 1:n]...], Int[])
    tensors = vcat([hmm.p0], [hmm.A for _ in 1:n-1], [hmm.B[:, o] for o in observations])
    optcode = optimize_code(code, OMEinsum.get_size_dict(code.ixs, (tensors...,)), optimizer)
    return HMMNetwork(n, optcode, observations, tensors)
end
likelihood(tnet::HMMNetwork) = tnet.code(tnet.tensors...)[]
function likelihood_and_gradient(tnet::HMMNetwork)
    likelihood, gradients = cost_and_gradient(tnet.code, (tnet.tensors...,))
    return likelihood[], gradients
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
        [tnet.tensors[ia][path[i], path[i+1]] for (i,ia) in enumerate(arange(tnet))],
        [tnet.tensors[ib][path[i]] for (i,ib) in enumerate(brange(tnet))]
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
    
    prev_likelihood = -Inf
    net = HMMNetwork(hmm, observations)
    for _ in 1:max_iter
        # E-step: Compute forward and backward probabilities
        likelihood, gradients = likelihood_and_gradient(net)
        ξ = [gradients[ia] ./ likelihood .* net.tensors[ia] for ia in arange(net)]  # transition probabilities
        γ = [gradients[ib] ./ likelihood .* net.tensors[ib] for ib in brange(net)]  # emission probabilities

        # Check convergence
        abs(likelihood - prev_likelihood) < tol && break
        prev_likelihood = likelihood
        
        # # M-step: Update parameters
        # Update initial state distribution
        new_p0 = γ[1]
        
        # Update transition matrix
        A = sum(ξ) ./ sum(γ[1:end-1])

        # Update emission matrix
        B = zeros(n_states, n_observations)
        for i in eachindex(γ)
            B[:, observations[i]] .+= γ[i]
        end
        B ./= sum(γ)
        # Create new HMM network with updated parameters
        net.tensors[arange(net)] .= Ref(A)
        net.tensors[brange(net)] .= [B[:, o] for o in observations]
        net.tensors[p0index(net)] = new_p0
    end
    
    return HMM(A, B, p0)
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