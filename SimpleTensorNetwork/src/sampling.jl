# A simple implementation of the MPS Born Machine
# Pan Zhang
# Institute of Theoretical Physics, Chinese Academy of Sciences
# Reference: Z. Han, J. Wang, H. Fan, L. Wang and P. Zhang, Phys. Rev. X 8, 031012 (2018)

function load_mnist()
    return npzread(joinpath(@__DIR__, "mnist784_bin_1000.npy"))
end

struct MPS{T}
    tensors::Vector{Array{T, 3}}
    function MPS(tensors::Vector{Array{T, 3}}) where T
        # the first and last tensor must be 1
        @assert size(tensors[1], 1) == 1
        @assert size(tensors[end], 3) == 1
        # the bond dimensions match
        @assert all(lr -> size(lr[1], 3) == size(lr[2], 1), zip(tensors[1:end-1], tensors[2:end]))
        new{T}(tensors)
    end
end
bond_dims(mps::MPS) = [size(t, 3) for t in mps.tensors]

function random_mps(::Type{T}, n::Int, Dmax::Int) where T
    tensors = [randn(T, (i == 1 ? 1 : Dmax), 2, (i == n ? 1 : Dmax)) for i in 1:n]
    return MPS(tensors)
end

# orthogonalize the MPS at a given site
function orthogonalize!(mps::MPS, site::Int, going_right::Bool)
    if going_right
        TA, TB = mps.tensors[site], mps.tensors[site+1]
        A = reshape(TA, :, size(TA, 3))
        Q, R = qr(A)
        R ./= norm(R)  # to avoid numerical instability
        mps.tensors[site] = reshape(Matrix(Q), size(TA, 1), 2, :)
        mps.tensors[site+1] = ein"ij,jkl->ikl"(R, TB)
    else
        TA, TB = mps.tensors[site], mps.tensors[site-1]
        A = reshape(TA, size(TA, 1), :)
        Q, R = qr(A')
        R ./= norm(R)
        mps.tensors[site] = reshape(Matrix(Q)', :, 2, size(TA, 3))
        mps.tensors[site-1] = ein"jkl,ml->jkm"(TB, R)
    end
end

# left canonicalize the MPS by sweeping from left to right
function left_canonicalize!(mps::MPS)
    n = length(mps.tensors)
    for site in 1:n-1
        orthogonalize!(mps, site, true)
    end
    return mps
end

# right canonicalize the MPS by sweeping from right to left
function right_canonicalize!(mps::MPS)
    n = length(mps.tensors)
    for site in n:-1:2
        orthogonalize!(mps, site, false)
    end
    return mps
end

# get the amplitude of given data
function get_psi(mps::MPS, data)
    m, n = size(data)
    psi = ones(Float32, 1, m, 1)
    for site in 1:n
        psi = ein"ibj,jbk->ibk"(psi, mps.tensors[site][:, data[:,site].+1, :])
    end
    return psi
end

# generate samples from the MPS
function gen_samples(mps::MPS, ns::Int)
    n = length(mps.tensors)
    # left canonicalize the MPS
    mps = deepcopy(mps)
    left_canonicalize!(mps)

    samples = zeros(Int, ns, n)
    for s in 1:ns
        vec = ones(Float32, 1)
        for site in n:-1:2
            vec = ein"imj,j->im"(mps.tensors[site], vec)
            p0 = norm(vec[:,1])^2 / norm(vec)^2
            x = rand() < p0 ? 0 : 1
            vec = vec[:,x+1]
            samples[s,site] = x
        end
    end
    return samples
end

function create_cache(mps::MPS, data)
    m, n = size(data)
    cache = []
    println("Caching...")
    push!(cache, ones(Float32, 1, m, 1)) # Initial element
    for site in 1:n-1
        B = ein"ibj,jbk->ibk"(cache[site], mps.tensors[site][:, data[:,site].+1, :])
        B ./= maximum(abs.(B))
        push!(cache, B)
    end
    push!(cache, ones(Float32, 1, m, 1)) # Last element
    return cache
end

function train(mps::MPS{T}, data, learning_rate=0.08f0, epochs=9) where T
    mps = deepcopy(mps)
    cache = create_cache(mps, data)
    m, n = size(data)
    results = MPS{T}[]
    for epoch in 1:epochs    # one sweep
        going_right = false
        t0 = time()
        for site in vcat(n:-1:2, 1:n-1)
            println("\r Epoch #$epoch, site #$site / $n           ")
            site == 1 && (going_right = true)
            
            gradients = zeros(Float32, size(mps.tensors[site]))
            for i in [0, 1]
                idx = findall(x -> x == i, data[:,site])
                isempty(idx) && continue
                
                left_vec = cache[site][:,idx,:]
                right_vec = cache[site+1][:,idx,:]
                A = mps.tensors[site][:, data[idx,site] .+ 1, :]
                
                psi = ein"(ibj,jbk),kbl->ibl"(left_vec, A, right_vec)
                gradients[:,i+1,:] = dropdims(sum(ein"ibj,kbi->jbk"(left_vec, right_vec) ./ psi, dims=2), dims=2)
            end
            
            gradients .= 2.0f0 .* (gradients ./ m .- mps.tensors[site])
            mps.tensors[site] .+= learning_rate .* gradients ./ norm(gradients)
            @show size(mps.tensors[site])
            orthogonalize!(mps, site, going_right)
            
            if going_right
                cache[site+1] = ein"ibj,jbk->ibk"(cache[site], mps.tensors[site][:, data[:,site].+1, :])
                cache[site+1] ./= maximum(abs.(cache[site+1]))
            else
                cache[site] = ein"ibj,jbk->ibk"(mps.tensors[site][:, data[:,site].+1, :], cache[site+1])
                cache[site] ./= maximum(abs.(cache[site]))
            end
        end
        psi = get_psi(mps, data)
        @info("NLL = $(round(-sum(log.(psi.^2))/length(psi), digits=3)), " *
                "LowerBound = $(round(log(m), digits=3)), " *
                "total_prob = $(round(sum(psi.^2), digits=3)) "
                )
        push!(results, deepcopy(mps))
    end
    return results
end

function plot_distribution(mps::MPS, data)
    psi = get_psi(mps, data)
    fig = Figure(size=(500, 300))
    ax = Axis(fig[1,1])
    barplot!(ax, 1:length(psi), dropdims(psi.^2, dims=(1,3)))
    fig
end

function generate_images(mps::MPS)
    imgs = gen_samples(mps, 30)
    show_imgs(imgs, 2, 15)
end
 
function show_imgs(imgs, nrow=4, ncol=5)
    imgs = reshape(imgs, :, 28, 28)
    fig = Figure(; size=(ncol*100, nrow*100))
    for i in 1:nrow
        for j in 1:ncol
            a = (i-1)*ncol + j
            if a > size(imgs, 1)
                break
            end
            ax = Axis(fig[i,j], aspect=1)
            heatmap!(ax, imgs[a,:,end:-1:1], colormap=:summer)
            hidedecorations!(ax)
            hidespines!(ax)
        end
    end
    fig
end

