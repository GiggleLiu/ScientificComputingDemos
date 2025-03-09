module LatticeBoltzmannCUDAExt
using CUDA: @kernel, get_backend, @index
using LatticeBoltzmannModel: Cell, AbstractLBConfig, directions, flip_direction_index, density, LatticeBoltzmann
using LatticeBoltzmannModel

function LatticeBoltzmannModel.stream!(lb::AbstractLBConfig{2, N}, newgrid::CuMatrix{D}, grid::CuMatrix{D}, barrier::CuMatrix{Bool}) where {N, T, D<:Cell{N, T}}
    ds = directions(lb)
    @kernel function kernel(newgrid, grid, barrier, ds)
        ci = @index(Global, Cartesian)
        i, j = ci.I
        @inbounds newgrid[ci] = Cell(ntuple(N) do k
            ei = ds[k]
            m, n = size(grid)
            i2, j2 = mod1(i - ei[1], m), mod1(j - ei[2], n)
            if barrier[i2, j2]
                density(grid[i, j], flip_direction_index(lb, k))
            else
                density(grid[i2, j2], k)
            end
        end)
    end
    kernel(get_backend(newgrid))(newgrid, grid, barrier, ds; ndrange=size(newgrid))
    return newgrid
end

function CUDA.cu(lb::LatticeBoltzmann{D, N}) where {D, N}
    return LatticeBoltzmann(lb.config, CuArray(lb.grid), CuArray(lb.barrier))
end
end
