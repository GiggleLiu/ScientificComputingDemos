using SimpleTensorNetwork

function asia_network(; open_vertices=Char[])
    # 0 -> NO
    # 1 -> YES
    factors = [
               # prior distributions of A and S
               "A" => [0.2, 0.8],
               "S" => [0.7, 0.3],
               # conditional probability tables
               "AT" => [0.98 0.02;
                        0.95 0.05],
               "EX" => [0.99 0.01;
                        0.02 0.98],
               "SB" => [0.96 0.04;
                        0.88 0.12],
               "SL" => [0.99 0.01;
                        0.92 0.08],
               "TLE" => (x = zeros(2, 2, 2);
                        x[1,:,:] .= [1.0 0.0;
                                    0.0 1.0];
                        x[2,:,:] .= [0.0 1.0;
                                    0.0 1.0]; x),
               "EBD" => (x = zeros(2, 2, 2);
                        x[1,:,:] .= [0.8 0.2;
                                    0.3 0.7];
                        x[2,:,:] .= [0.2 0.8;
                                    0.05 0.95]; x)]
    return TensorNetwork(getfield.(factors, :second), collect.(first.(factors)), open_vertices)
end

network = asia_network()
optnet = optimize_tensornetwork(network)
total_prob = optnet.ein(network.tensors...)[]

network = asia_network()


# Import the TensorInference package, which provides the functionality needed
# for working with tensor networks and probabilistic graphical models.
using TensorInference

# ---

# Load the ASIA network model from the `asia.uai` file located in the examples
# directory. See [Model file format (.uai)](@ref) for a description of the
# format of this file.
model = read_model_file(pkgdir(TensorInference, "examples", "asia-network", "model.uai"))

# ---

# Create a tensor network representation of the loaded model.
tn = TensorNetworkModel(model)

# ---

# Calculate the partition function. Since the factors in this model are
# normalized, the partition function is the same as the total probability, $1$.
probability(tn) |> first

# ---

# Calculate the marginal probabilities of each random variable in the model.
marginals(tn)

# ---

# Retrieve all the variables in the model.
get_vars(tn)

# ---

# Set the evidence: Assume that the "X-ray" result (variable 7) is negative.
# Since setting the evidence may affect the contraction order of the tensor
# network, recompute it.
tn = TensorNetworkModel(model, evidence = Dict(7 => 0))

# ---

# Calculate the maximum log-probability among all configurations.
maximum_logp(tn)

# ---

# Generate 10 samples from the posterior distribution.
sample(tn, 10)

# ---

# Retrieve both the maximum log-probability and the most probable
# configuration. 
logp, cfg = most_probable_config(tn)

# ---

# Compute the most probable values of certain variables (e.g., 4 and 7) while
# marginalizing over others. This is known as Maximum a Posteriori (MAP)
# estimation.
mmap = MMAPModel(model, evidence=Dict(7=>0), queryvars=[4,7])

# ---

# Get the most probable configurations for variables 4 and 7.
most_probable_config(mmap)

# ---

# Compute the total log-probability of having lung cancer. The results suggest
# that the probability is roughly half.
log_probability(mmap, [1, 0]), log_probability(mmap, [0, 0])