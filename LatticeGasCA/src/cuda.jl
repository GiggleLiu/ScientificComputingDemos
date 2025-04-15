function update!(lg::HPPLatticeGas{ET, <:CuArray{ET}}) where ET
    @kernel function kernel(lattice::AbstractArray{T}, cache) where T
        i, j = @index(Global, NTuple)
        nx, ny = size(lattice)
        @inbounds state = (i == nx ? zero(T) : left(lattice[i+1, j])) +
            (i == 1 ? zero(T) : right(lattice[i-1, j])) + 
            (j == ny ? zero(T) : down(lattice[i, j+1])) +
            (j == 1 ? zero(T) : up(lattice[i, j-1]))
        newstate = hpp_state_transfer_rule(state, i, j, nx, ny)
        @inbounds cache[i, j] = newstate
    end
    backend = get_backend(lg.lattice)
    CUDA.@sync kernel(backend)(lg.lattice, lg.cache; ndrange=size(lg.lattice))
    copyto!(lg.lattice, lg.cache)
    return lg
end

function CUDA.cu(lg::HPPLatticeGas)
    return HPPLatticeGas(CuArray(lg.lattice), CuArray(lg.cache))
end

function cpu(lg::HPPLatticeGas)
    return HPPLatticeGas(Array(lg.lattice), Array(lg.cache))
end