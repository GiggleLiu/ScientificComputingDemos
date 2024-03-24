# A simple spinglass model:
# H = -\sum_{i,j} J_{ij} \sigma_i \sigma_j + \sum_i h_i \sigma_i
struct Spinglass{T}
    graph::SimpleGraph{Int}
    J::Vector{T}
    h::Vector{T}
end

struct TensorNetwork{T, LT}
    tensors::AbstractArray{T}
    ixs::Vector{Vector{LT}}
    iy::Vector{LT}
end

struct OptimizedTensorNetwork{T, ET<:AbstractEinsum}
    tensors::AbstractArray{T}
    ein::ET
end

function generate_tensor_network(sg::Spinglass{T}, β; output_indices = Int[]) where T
    tensors = AbstractArray[]
    ixs = Vector{Int}[]
    for (e, Jij) in zip(edges(sg.graph), sg.J)
        push!(tensors, spinglass_edgetensor(Jij, T(β)))
        push!(ixs, [src(e), dst(e)])
    end
    for (v, hi) in zip(vertices(sg.graph), sg.h)
        push!(tensors, spinglass_vertextensor(hi, T(β)))
        push!(ixs, [v])
    end
    return TensorNetwork(tensors, ixs, output_indices)
end
spinglass_edgetensor(J::T, β::T) where T = exp.(-β .* [-J J; J -J])
spinglass_vertextensor(h::T, β::T) where T = exp.(-β .* [h, -h])

function optimize_tensornetwork(tnet::TensorNetwork; optimizer=TreeSA())
    eincode = DynamicEinCode(tnet.ixs, tnet.iy)
    optimized_eincode = optimize_code(eincode, uniformsize(eincode, 2), optimizer, MergeVectors())
    return OptimizedTensorNetwork(tnet.tensors, optimized_eincode)
end

function partition_function(sg::Spinglass, β; optimizer=TreeSA())
    @debug "generating the tensor network"
    tnet = generate_tensor_network(sg, β)
    @debug "optimize the contraction order with: $optimizer"
    optnet = optimize_tensornetwork(tnet; optimizer)
    @debug "contract the tensor network"
    return optnet.ein(tnet.tensors...)[]
end

function partition_function_exact(sg::Spinglass, β)
    Z = 0.0
    for σ in Iterators.product(fill([-1, 1], nv(sg.graph))...)
        E = 0.0
        for (e, Jij) in zip(edges(sg.graph), sg.J)
            srcv, dstv = src(e), dst(e)
            E -= Jij * σ[srcv] * σ[dstv]
        end
        for (v, hi) in zip(vertices(sg.graph), sg.h)
            E += hi * σ[v]
        end
        Z += exp(-β * E)
    end
    return Z
end