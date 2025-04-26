module CUDAExt

using CUDA, Spinglass
using Spinglass: SpinGlassSA
using CUDA.GPUArrays: @kernel, get_backend, @index

# upload the coupling matrix to the GPU
CUDA.cu(sa::SpinGlassSA) = SpinGlassSA(CUDA.CuArray(sa.coupling))
cpu(sa::SpinGlassSA) = SpinGlassSA(Matrix(sa.coupling))

struct BatchedSpinConfig{T1, T2, MT1<:AbstractMatrix{T1}, MT2<:AbstractMatrix{T2}}
    config::MT1
    field::MT2
end

function Spinglass.anneal(nrun::Int, prob::SpinGlassSA{TF, <:CuMatrix{TF}}, tempscales::CuVector{TF}, num_update_each_temp::Int) where {TF}
    initial_config = [random_config(cpu(prob)) for _ in 1:nrun]
    batch_config = BatchedSpinConfig(CUDA.CuArray(hcat(getfield.(initial_config, :config)...)), CUDA.CuArray(hcat(getfield.(initial_config, :field)...)))
    anneal_run!(batch_config, prob, tempscales, num_update_each_temp)
    cpu_config = BatchedSpinConfig(Matrix(batch_config.config), Matrix(batch_config.field))
    eng, idx = findmin(i -> Spinglass.energy(SpinConfig(cpu_config.config[:, i], cpu_config.field[:, i]), cpu(prob)), 1:nrun)
    return eng, SpinConfig(cpu_config.config[:, idx], cpu_config.field[:, idx])
end

function anneal_run!(config::BatchedSpinConfig{TI, TF, <:CuMatrix{TI}, <:CuMatrix{TF}}, prob::SpinGlassSA{TF, <:CuMatrix{TF}}, tempscales::CuVector{TF}, num_update_each_temp::Int) where {TI, TF}
   @kernel function kernel(config, field, coupling)
        ibatch = @index(Global, Linear)
        for temp in tempscales
            beta = inv(temp)
            for _ = 1:num_update_each_temp  # single instriuction multiple data, see julia performance tips.
                proposal, ΔE = propose(config, field, coupling, ibatch)
                if exp(-beta*ΔE) > CUDA.Random.rand()  #accept
                    flip!(config, field, proposal, coupling, ibatch)
                end
            end
        end
    end
    # we only parallel over the batch size, not the spin number.
    kernel(get_backend(config.config))(config.config, config.field, prob.coupling; ndrange=size(config.config, 2))
end
 

@inline function propose(config, field, coupling, ibatch::Int)  # ommit the name of argument, since not used.
    ispin = CUDA.Random.rand(1:size(coupling, 1))
    ΔE = -field[ispin, ibatch] * config[ispin, ibatch] * 4 # 2 for spin change, 2 for mutual energy.
    ispin, ΔE
end

@inline function flip!(config, field, ispin::Int, coupling, ibatch::Int)
    @inbounds config[ispin, ibatch] = -config[ispin, ibatch]  # @inbounds can remove boundary check, and improve performance
    # update the field
    for i=1:size(coupling, 1)
        @inbounds field[i, ibatch] += 2 * config[ispin, ibatch] * coupling[i,ispin]
    end
end

@info "`CUDAExt` (for `Spinglass`) is loaded successfully."

end
